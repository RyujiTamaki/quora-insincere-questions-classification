from contextlib import contextmanager
import gc
import os
import time
import numpy as np
import pandas as pd
import joblib
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


MAX_FEATURES = 50000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300
GLOVE_PATH = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
FAST_TEXT_PATH = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
PARAGRAM_PATH = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
WORD2VEC_PATH = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def load_embedding_matrix(word_index,
                          embedding_path=None):
    with timer('load embeddings'):
        num_words = min(MAX_FEATURES, len(word_index)) + 1
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        embeddings_index = {}
        if embedding_path == GLOVE_PATH:
            f = open(embedding_path)
            for line in f:
                values = line.split(" ")
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()

        if embedding_path == FAST_TEXT_PATH:
            f = open(embedding_path)
            for line in f:
                values = line.split(" ")
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()

        if embedding_path == PARAGRAM_PATH:
            f = open(embedding_path, encoding="utf8", errors='ignore')
            for line in f:
                values = line.split(" ")
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()

        if embedding_path == WORD2VEC_PATH:
            embeddings_index = KeyedVectors.load_word2vec_format(
                embedding_path, binary=True)

        for word, i in word_index.items():
            if i >= MAX_FEATURES:
                continue
            if embedding_path == WORD2VEC_PATH:
                try:
                    embedding_vector = embeddings_index.get_vector(word)
                except KeyError:
                    pass
            else:
                embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


def main():
    with timer('load data'):
        train_df = pd.read_csv("../input/train.csv")
        test_df = pd.read_csv('../input/test.csv')
        X_train = train_df["question_text"].fillna("_na_").values
        y_train = train_df["target"].values
        X_test = test_df["question_text"].fillna("_na_").values

    with timer('preprocess'):
        tokenizer = Tokenizer(num_words=MAX_FEATURES)
        tokenizer.fit_on_texts(np.hstack((
            train_df.question_text.values, test_df.question_text.values))
        )
        X_train = tokenizer.texts_to_sequences(train_df.question_text.values)
        X_test = tokenizer.texts_to_sequences(test_df.question_text.values)
        word_index = tokenizer.word_index

        X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
        X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

    glove_embedding = load_embedding_matrix(
        word_index=word_index,
        embedding_path=GLOVE_PATH
    )

    fast_text_embedding = load_embedding_matrix(
        word_index=word_index,
        embedding_path=FAST_TEXT_PATH
    )

    paragram_embedding = load_embedding_matrix(
        word_index=word_index,
        embedding_path=PARAGRAM_PATH
    )

    word2vec_embedding = load_embedding_matrix(
        word_index=word_index,
        embedding_path=WORD2VEC_PATH
    )

    with timer('dump data'):
        joblib.dump(X_train, '../input/X_train.joblib')
        joblib.dump(y_train, '../input/y_train.joblib')
        joblib.dump(X_test, '../input/X_test.joblib')
        joblib.dump(glove_embedding, '../input/glove_embedding.joblib')
        joblib.dump(fast_text_embedding, '../input/fast_text_embedding.joblib')
        joblib.dump(paragram_embedding, '../input/paragram_embedding.joblib')
        joblib.dump(word2vec_embedding, '../input/word2vec_embedding.joblib')


if __name__ == '__main__':
    main()
