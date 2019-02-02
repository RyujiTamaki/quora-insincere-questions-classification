# standard imports
import random
import re
import os
import time
import numpy as np
import pandas as pd

# pytorch imports
import torch
import torch.nn as nn
import torch.utils.data

# imports for preprocessing the questions
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# progress bars
from tqdm import tqdm
tqdm.pandas()


embed_size = 300  # how big is each word vector
max_features = 95000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70  # max number of words in a question to use
batch_size = 512  # how many samples to process at once
n_epochs = 5  # how many times to iterate over all samples
enable_local_test = True


class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, factor=0.6, min_lr=1e-4, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

        self.last_loss = np.inf
        self.min_lr = min_lr
        self.factor = factor

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def step(self, loss):
        if loss > self.last_loss:
            self.base_lrs = [max(lr * self.factor, self.min_lr) for lr in self.base_lrs]
            self.max_lrs = [max(lr * self.factor, self.min_lr) for lr in self.max_lrs]

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class NeuralNet(nn.Module):
    def __init__(self,
                 dropout_rate=0.1,
                 lstm_hidden_size=60,
                 last_hidden_size=16):
        super(NeuralNet, self).__init__()

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(dropout_rate)
        self.lstm = nn.LSTM(embed_size, lstm_hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(lstm_hidden_size * 2, lstm_hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(lstm_hidden_size * 2, maxlen)
        self.gru_attention = Attention(lstm_hidden_size * 2, maxlen)

        self.linear = nn.Linear(lstm_hidden_size * 8 + 2, last_hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(last_hidden_size, 1)

    def forward(self, x, f):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(
            self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)

        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)

        # global average pooling
        avg_pool = torch.mean(h_gru, 1)
        # global max pooling
        max_pool, _ = torch.max(h_gru, 1)

        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool, f), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out


def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def threshold_search(y_true, y_proba):
    args = np.argsort(y_proba)
    tp = y_true.sum()
    fs = (tp - np.cumsum(y_true[args[:-1]])) / np.arange(y_true.shape[0] + tp - 1, tp, -1)
    res_idx = np.argmax(fs)
    best_th = (y_proba[args[res_idx]] + y_proba[args[res_idx + 1]]) / 2
    best_score = 2 * fs[res_idx]
    search_result = {'threshold': best_th, 'f1': best_score}
    return search_result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clean_text(x, maxlen=None):
    puncts = [
        ',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£',
        '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
        '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
        '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
        '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√',
    ]
    x = x.lower()
    for punct in puncts[:maxlen]:
        if punct in x:  # add this line
            x = x.replace(punct, f' {punct} ')
    return x


def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x


mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispellings, mispellings_re = _get_mispell(mispell_dict)


def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)


def load_glove(word_index, max_words=200000, embed_size=300):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    emb_mean, emb_std = -0.005838499, 0.48782197

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_words, embed_size))
    with open(EMBEDDING_FILE, 'r', encoding="utf8") as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word not in word_index:
                continue
            i = word_index[word]
            if i >= max_words:
                continue
            embedding_vector = np.asarray(vec.split(' '), dtype='float32')[:300]
            if len(embedding_vector) == 300:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


def load_para(word_index, max_words=200000, embed_size=300):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    emb_mean, emb_std = -0.0053247833, 0.49346462

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_words, embed_size))
    with open(EMBEDDING_FILE, 'r', encoding="utf8", errors='ignore') as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word not in word_index:
                continue
            i = word_index[word]
            if i >= max_words:
                continue
            embedding_vector = np.asarray(vec.split(' '), dtype='float32')[:300]
            if len(embedding_vector) == 300:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_fast(word_index, max_words=200000, embed_size=300):
    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    emb_mean, emb_std = -0.0033469985, 0.109855495

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_words, embed_size))
    with open(EMBEDDING_FILE, 'r', encoding="utf8") as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word not in word_index:
                continue
            i = word_index[word]
            if i >= max_words:
                continue
            embedding_vector = np.asarray(vec.split(' '), dtype='float32')[:300]
            if len(embedding_vector) == 300:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


def train_model(model, x_train, x_feature_train, y_train, x_val, x_feature_val, y_val, validate=True):

    train = torch.utils.data.TensorDataset(x_train, x_feature_train, y_train)
    valid = torch.utils.data.TensorDataset(x_val, x_feature_val, y_val)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters())
    step_size = 300
    scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.003,
                         step_size=step_size, mode='exp_range',
                         gamma=0.99994)

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()
    best_score = -np.inf

    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.

        for x_batch, f_batch, y_batch in tqdm(train_loader, disable=True):
            y_pred = model(x_batch, f_batch)
            scheduler.batch_step()
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        model.eval()
        valid_preds = np.zeros((x_val_fold.size(0)))

        if validate:
            avg_val_loss = 0.
            for i, (x_batch, f_batch, y_batch) in enumerate(valid_loader):
                y_pred = model(x_batch, f_batch).detach()

                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
            search_result = threshold_search(y_val.cpu().numpy(), valid_preds)
            auc_score = roc_auc_score(y_val.cpu().numpy(), valid_preds)

            val_f1, val_threshold = search_result['f1'], search_result['threshold']
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t val_f1={:.4f} \t val_auc={:.4f} \t best_t={:.2f} \t time={:.2f}s'.format(
                epoch + 1, n_epochs, avg_loss, avg_val_loss, val_f1, auc_score, val_threshold, elapsed_time))
        else:
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, n_epochs, avg_loss, elapsed_time))

    valid_preds = np.zeros((x_val_fold.size(0)))

    avg_val_loss = 0.
    for i, (x_batch, f_batch, y_batch) in enumerate(valid_loader):
        y_pred = model(x_batch, f_batch).detach()

        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        valid_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    print('Validation loss: ', avg_val_loss)

    test_preds = np.zeros((len(test_loader.dataset)))

    for i, (x_batch, f_batch) in enumerate(test_loader):
        y_pred = model(x_batch, f_batch).detach()

        test_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    test_preds_local = np.zeros((len(test_local_loader.dataset)))

    for i, (x_batch, f_batch) in enumerate(test_local_loader):
        y_pred = model(x_batch, f_batch).detach()

        test_preds_local[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    return valid_preds, test_preds, test_preds_local


def add_features(df):
    df['question_text'] = df['question_text'].apply(lambda x: str(x))
    df['total_length'] = df['question_text'].apply(len)
    df['capitals'] = df['question_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals']) / float(row['total_length']), axis=1)
    df['num_words'] = df.question_text.str.count('\S+')
    df['num_unique_words'] = df['question_text'].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']

    return df


seed_torch()

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


if enable_local_test:
    n_test = len(test_df) * 4
    train_df, local_test_df = (train_df.iloc[:-n_test].reset_index(drop=True),
                               train_df.iloc[-n_test:].reset_index(drop=True))
else:
    local_test_df = pd.DataFrame([[None, "", 0], [None, "", 0]], columns=['qid', 'question_text', 'target'])
    n_test = 2

# Add Features
train = add_features(train_df)
test = add_features(test_df)
local_test = add_features(local_test_df)

x_feature_train = train[['caps_vs_length', 'words_vs_unique']].fillna(0)
x_feature_test = test[['caps_vs_length', 'words_vs_unique']].fillna(0)
x_feature_local_test = local_test[['caps_vs_length', 'words_vs_unique']].fillna(0)

ss = StandardScaler()
ss.fit(np.vstack((x_feature_train, x_feature_test, x_feature_local_test)))
x_feature_train = ss.transform(x_feature_train)
x_feature_test = ss.transform(x_feature_test)
x_feature_local_test = ss.transform(x_feature_local_test)

# preprocess
for df in [train_df, test_df, local_test_df]:
    df["question_text"] = df["question_text"].str.lower()
    df["question_text"] = df["question_text"].apply(lambda x: clean_text(x))
    # df["question_text"] = df["question_text"].apply(lambda x: clean_numbers(x))
    # df["question_text"] = df["question_text"].apply(lambda x: replace_typical_misspell(x))
    df["question_text"].fillna("_##_", inplace=True)

x_train = train_df["question_text"].values
x_test = test_df["question_text"].values
x_test_local = local_test_df["question_text"].values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_train) + list(x_test_local))
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_test_local = tokenizer.texts_to_sequences(x_test_local)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
x_test_local = pad_sequences(x_test_local, maxlen=maxlen)

y_train = train_df['target'].values
y_test = local_test_df['target'].values

glove_embeddings = load_glove(tokenizer.word_index, len(tokenizer.word_index) + 1)
paragram_embeddings = load_para(tokenizer.word_index, len(tokenizer.word_index) + 1)
# fasttext_embeddings = load_fast(tokenizer.word_index, len(tokenizer.word_index) + 1)

embedding_matrix = np.mean([glove_embeddings, paragram_embeddings], axis=0)
np.shape(embedding_matrix)

splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=10).split(x_train, y_train))
train_preds = np.zeros((len(train_df)))
test_preds = np.zeros((len(test_df)))

x_test_cuda = torch.tensor(x_test, dtype=torch.long).cuda()
x_feature_test_cuda = torch.tensor(x_feature_test, dtype=torch.float).cuda()
test = torch.utils.data.TensorDataset(x_test_cuda, x_feature_test_cuda)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

x_test_local_cuda = torch.tensor(x_test_local, dtype=torch.long).cuda()
x_feature_local_test_cuda = torch.tensor(x_feature_local_test, dtype=torch.float).cuda()
test_local = torch.utils.data.TensorDataset(x_test_local_cuda, x_feature_local_test_cuda)
test_local_loader = torch.utils.data.DataLoader(test_local, batch_size=batch_size, shuffle=False)

train_preds = np.zeros(len(train_df))
test_preds = np.zeros((len(test_df), len(splits)))
test_preds_local = np.zeros((n_test, len(splits)))

seed = 6017

for i, (train_idx, valid_idx) in enumerate(splits):
    x_train_fold = torch.tensor(x_train[train_idx], dtype=torch.long).cuda()
    y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32).cuda()
    x_val_fold = torch.tensor(x_train[valid_idx], dtype=torch.long).cuda()
    y_val_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32).cuda()
    x_feature_train_fold = torch.tensor(x_feature_train[train_idx], dtype=torch.float).cuda()
    x_feature_val_fold = torch.tensor(x_feature_train[valid_idx], dtype=torch.float).cuda()

    print(f'Fold {i + 1}')

    seed_torch(seed + i)
    model = NeuralNet()
    model.cuda()

    valid_preds_fold, test_preds_fold, test_preds_local_fold = train_model(
        model,
        x_train_fold,
        x_feature_train_fold,
        y_train_fold,
        x_val_fold,
        x_feature_val_fold,
        y_val_fold,
        validate=True
    )

    train_preds[valid_idx] = valid_preds_fold
    test_preds[:, i] = test_preds_fold
    test_preds_local[:, i] = test_preds_local_fold

search_result = threshold_search(y_train, train_preds)
print("search result: {}".format(search_result))
print("valid_preds_local auc: {}".format(roc_auc_score(y_train, train_preds)))
print("pd.DataFrame(test_preds_local).corr()")
print(pd.DataFrame(test_preds_local).corr())
print("f1_score(y_test, test_preds_local.mean(axis=1) > search_result['threshold'])")
print(f1_score(y_test, test_preds_local.mean(axis=1) > search_result['threshold']))

print("test_preds_local auc: {}".format(roc_auc_score(y_test, test_preds_local.mean(axis=1))))

submission = test_df[['qid']].copy()
submission['prediction'] = test_preds.mean(axis=1) > search_result['threshold']
submission.to_csv('submission.csv', index=False)
