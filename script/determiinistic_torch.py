from contextlib import contextmanager
import gc
import os
import time
import numpy as np
import pandas as pd
import random as rn
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import torch.utils.data
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf


max_features = 95000
maxlen = 70
embed_size = 300
batch_size = 145
n_epochs = 3
NFOLDS = 4
GLOVE_PATH = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
FAST_TEXT_PATH = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
PARAGRAM_PATH = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
WORD2VEC_PATH = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
puncts = [
    ',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#',
    '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',
    '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬',
    '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†',
    '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓',
    '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√'
]


def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def seed_torch(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')[:300]

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.005838499, 0.48782197
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_para(word_index):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.0053247833, 0.49346462
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


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
                 lstm_hidden_size=40,
                 last_hidden_size=16,
                 embedding_matrix=None):
        super(NeuralNet, self).__init__()

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(dropout_rate)
        self.lstm = nn.LSTM(embed_size, lstm_hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(lstm_hidden_size * 2, lstm_hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(lstm_hidden_size * 2, maxlen)
        self.gru_attention = Attention(lstm_hidden_size * 2, maxlen)

        self.linear = nn.Linear(lstm_hidden_size * 8, last_hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(last_hidden_size, 1)

    def forward(self, x):
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

        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out


def main():
    with timer('load data'):
        train_df = pd.read_csv("../input/train.csv")
        test_df = pd.read_csv("../input/test.csv")
        train_df["question_text"] = train_df["question_text"].str.lower()
        test_df["question_text"] = test_df["question_text"].str.lower()
        train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
        test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))
        X_train = train_df["question_text"].fillna("_na_").values
        y_train = train_df["target"].values
        X_test = test_df["question_text"].fillna("_na_").values
        qid = test_df["qid"]

    with timer('preprocess'):
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(
            list(X_train) + list(X_test)
        )
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)

        X_train = pad_sequences(X_train, maxlen=maxlen)
        X_test = pad_sequences(X_test, maxlen=maxlen)

    glove_embeddings = load_glove(tokenizer.word_index)
    paragram_embeddings = load_para(tokenizer.word_index)

    embedding_matrix = np.mean([glove_embeddings, paragram_embeddings], axis=0)

    skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=10)
    skf_y = np.zeros((len(y_train),))

    train_preds = np.zeros((len(train_df)))
    test_preds = np.zeros((len(test_df)))

    seed_torch()

    x_test_cuda = torch.tensor(X_test, dtype=torch.long).cuda()
    test = torch.utils.data.TensorDataset(x_test_cuda)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    with timer("fit and pred"):
        for i, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
            x_train_fold = torch.tensor(X_train[train_idx], dtype=torch.long).cuda()
            y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32).cuda()
            x_val_fold = torch.tensor(X_train[valid_idx], dtype=torch.long).cuda()
            y_val_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32).cuda()

            model = NeuralNet(
                dropout_rate=0.09447228172786953,
                lstm_hidden_size=102,
                last_hidden_size=26,
                embedding_matrix=embedding_matrix
            )
            model.cuda()
            loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005089749049450687, weight_decay=1.1219048364147381e-10)

            train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
            valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

            print(f'Fold {i + 1}')

            for epoch in range(n_epochs):
                start_time = time.time()
                model.train()
                avg_loss = 0.
                for x_batch, y_batch in train_loader:

                    y_pred = model(x_batch)
                    loss = loss_fn(y_pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item() / len(train_loader)

                model.eval()

                valid_preds_fold = np.zeros((x_val_fold.size(0)))
                test_preds_fold = np.zeros((len(test_df)))

                avg_val_loss = 0.
                avg_f1 = 0.
                for i, (x_batch, y_batch) in enumerate(valid_loader):
                    y_pred = model(x_batch).detach()
                    y_proba = sigmoid(y_pred.cpu().numpy())[:, 0]
                    search_result = threshold_search(y_batch.cpu().numpy(), y_proba)
                    avg_f1 += search_result['f1'] / len(valid_loader)

                    avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                    valid_preds_fold[i * batch_size:(i + 1) * batch_size] = y_proba

                elapsed_time = time.time() - start_time
                print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t f1={:.4f} \t time={:.2f}s'.format(
                    epoch + 1, n_epochs, avg_loss, avg_val_loss, avg_f1, elapsed_time))

            for i, (x_batch,) in enumerate(test_loader):
                y_pred = model(x_batch).detach()

                test_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

            train_preds[valid_idx] = valid_preds_fold
            test_preds += test_preds_fold / NFOLDS

    search_result = threshold_search(y_train, train_preds)
    print("search result: {}".format(search_result))
    submission = test_df[['qid']].copy()
    submission['prediction'] = test_preds > search_result['threshold']
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
