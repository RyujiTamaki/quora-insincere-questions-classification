# standard imports
import gc
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

# cross validation and metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve

# SIF
from contextlib import contextmanager
import itertools
from collections import Counter
from sklearn.decomposition import TruncatedSVD

embed_size = 300  # how big is each word vector
max_features = 95000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70  # max number of words in a question to use
batch_size = 512  # how many samples to process at once
n_epochs = 4  # how many times to iterate over all samples


class SifEmbedding():
    def __init__(self, embeddings, a=1e-3):
        self.embeddings = embeddings
        self.embedding_dim = 300
        self.a = a
        self.word_counts = tokenizer.word_counts
        self.index2word = dict(map(reversed, tokenizer.word_index.items()))
        self.sentence_embedding = np.array([])

    def _weighted_bow(self, sentence):
        vs = np.zeros(self.embedding_dim)
        sentence_length = 0

        for word_idx in sentence:
            word = self.index2word.get(word_idx)
            if word is not None:
                a_value = self.a / (self.a + self.word_counts[word])  # smooth inverse frequency, SIF
                embedding_vector = self.embeddings[word_idx]
                if embedding_vector is not None:
                    vs = np.add(vs, np.multiply(a_value, embedding_vector))  # vs += sif * word_vector
                    sentence_length += 1

        if sentence_length != 0:
            vs = np.divide(vs, sentence_length)

        return vs

    def _fit_svd(self, X):
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
        svd.fit(X)
        self.u = svd.components_
        return self

    def _transform_svd(self, X):
        vs = X - X.dot(self.u.transpose()) * self.u
        return vs

    def _fit_transform_svd(self, X):
        return self._fit_svd(X)._transform_svd(X)

    def fit_transform(self, sentence_list):
        print("step 1")
        # Alg.1 step 1
        # sentence_vec = []
        sentence_vec = np.empty((len(sentence_list), self.embedding_dim), dtype=np.float32)
        for i, sentence in enumerate(sentence_list):
            # sentence_vec.append(self._weighted_bow(sentence))
            sentence_vec[i] = self._weighted_bow(sentence)

        print("step 2")
        # Alg.1 step 2
        self.sentence_embedding = self._fit_transform_svd(sentence_vec)

        return self.sentence_embedding



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
                 dropout_rate=0.04892470394937945,
                 lstm_hidden_size=120,
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

        self.linear = nn.Linear(lstm_hidden_size * 8 + embed_size, last_hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(last_hidden_size, 1)

    def forward(self, x_tokens, x_sif):
        h_embedding = self.embedding(x_tokens)
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

        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool, x_sif), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out


def seed_torch(seed=1234):
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


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
submission = test_df[['qid']].copy()
train_num = train_df.shape[0]
test_num = test_df.shape[0]

train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x, maxlen=maxlen))
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x, maxlen=maxlen))

# fill up the missing values
x_train = train_df["question_text"].fillna("_##_").values
x_test = test_df["question_text"].fillna("_##_").values

del train_df, test_df
gc.collect()

# Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_train) + list(x_test))
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

y_train = train_df['target'].values

glove_embeddings = load_glove(tokenizer.word_index, len(tokenizer.word_index) + 1)
paragram_embeddings = load_para(tokenizer.word_index, len(tokenizer.word_index) + 1)
embedding_matrix = np.mean([glove_embeddings, paragram_embeddings], axis=0)

# SIF
print('SifEmbedding')
sif = SifEmbedding(embeddings=paragram_embeddings)
x_sif = sif.fit_transform(np.concatenate([x_train, x_test]))
x_sif_train = x_sif[:train_num]
x_sif_test = x_sif[train_num:]
del sif, x_sif
gc.collect()
print('done')

splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=10).split(x_train, y_train))
train_preds = np.zeros((train_num))
test_preds = np.zeros((test_num))
seed_torch()

x_test_cuda = torch.tensor(x_test, dtype=torch.long).cuda()
x_test_sif = torch.tensor(x_sif_test, dtype=torch.float).cuda()
test = torch.utils.data.TensorDataset(x_test_cuda, x_test_sif)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

for i, (train_idx, valid_idx) in enumerate(splits):
    x_train_fold = torch.tensor(x_train[train_idx], dtype=torch.long).cuda()
    y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32).cuda()
    x_val_fold = torch.tensor(x_train[valid_idx], dtype=torch.long).cuda()
    y_val_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32).cuda()

    x_sif_train_fold = torch.tensor(x_sif_train[train_idx], dtype=torch.float).cuda()
    x_sif_val_fold = torch.tensor(x_sif_train[valid_idx], dtype=torch.float).cuda()

    model = NeuralNet()
    model.cuda()
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000873264229417701)

    train = torch.utils.data.TensorDataset(x_train_fold, x_sif_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_val_fold, x_sif_val_fold,  y_val_fold)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    print(f'Fold {i + 1}')

    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.
        for x_batch, x_sif_batch, y_batch in train_loader:

            y_pred = model(x_batch, x_sif_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        model.eval()

        valid_preds_fold = np.zeros((x_val_fold.size(0)))
        test_preds_fold = np.zeros((test_num))

        avg_val_loss = 0.
        avg_f1 = 0.
        for i, (x_batch, x_sif_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch, x_sif_batch).detach()
            y_proba = sigmoid(y_pred.cpu().numpy())[:, 0]
            search_result = threshold_search(y_batch.cpu().numpy(), y_proba)
            avg_f1 += search_result['f1'] / len(valid_loader)

            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            valid_preds_fold[i * batch_size:(i + 1) * batch_size] = y_proba

        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t f1={:.4f} \t time={:.2f}s'.format(
            epoch + 1, n_epochs, avg_loss, avg_val_loss, avg_f1, elapsed_time))

    for i, (x_batch, x_sif_batch, ) in enumerate(test_loader):
        y_pred = model(x_batch, x_sif_batch).detach()

        test_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    train_preds[valid_idx] = valid_preds_fold
    test_preds += test_preds_fold / len(splits)

search_result = threshold_search(y_train, train_preds)
print("search result: {}".format(search_result))
submission['prediction'] = test_preds > search_result['threshold']
submission.to_csv('submission.csv', index=False)
