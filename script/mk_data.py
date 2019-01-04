# standard imports
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

# progress bars
from tqdm import tqdm
tqdm.pandas()

import joblib


embed_size = 300  # how big is each word vector
max_features = 95000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70  # max number of words in a question to use
batch_size = 512  # how many samples to process at once
n_epochs = 6  # how many times to iterate over all samples


def seed_torch(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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

train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x, maxlen=maxlen))
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x, maxlen=maxlen))

# fill up the missing values
x_train = train_df["question_text"].fillna("_##_").values
x_test = test_df["question_text"].fillna("_##_").values

# Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_train))
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

y_train = train_df['target'].values

glove_embeddings = load_glove(tokenizer.word_index, len(tokenizer.word_index) + 1)
paragram_embeddings = load_para(tokenizer.word_index, len(tokenizer.word_index) + 1)

joblib.dump(x_train, '../input/x_train.joblib')
joblib.dump(y_train, '../input/y_train.joblib')
joblib.dump(x_test, '../input/x_test.joblib')
joblib.dump(glove_embeddings, '../input/glove_embeddings.joblib')
joblib.dump(paragram_embeddings, '../input/paragram_embeddings.joblib')
