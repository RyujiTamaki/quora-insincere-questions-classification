from contextlib import contextmanager
import gc
import os
import time
import numpy as np
import pandas as pd
import random as rn
from gensim.models import KeyedVectors
from keras import backend as K
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.initializers import *
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
import tensorflow as tf


# os.environ['PYTHONHASHSEED'] = '0'
# np.random.seed(42)
# rn.seed(12345)

MAX_FEATURES = 95000
MAX_SEQUENCE_LENGTH = 70
EMBEDDING_DIM = 300
NFOLDS = 4
GLOVE_PATH = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
FAST_TEXT_PATH = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
PARAGRAM_PATH = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
WORD2VEC_PATH = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


def load_embedding_matrix(word_index,
                          embedding_path=None):
    with timer('load embeddings'):
        num_words = min(MAX_FEATURES, len(word_index)) + 1
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

        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embedding_matrix = np.random.normal(emb_mean, emb_std, (num_words, EMBEDDING_DIM))

        for word, i in word_index.items():
            if i >= MAX_FEATURES:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


def build_gru(hidden_dim,
              dropout_rate,
              input_shape,
              model_type=0,
              is_embedding_trainable=False,
              meta_embeddings='DME',
              embedding_matrix=None):
    inp = Input(shape=(input_shape[0],))
    embeddings = []
    # Dynamic Meta-Embeddings for Improved Sentence Representations
    # https://arxiv.org/abs/1804.07983
    if meta_embeddings == 'concat':
        for weights in embedding_matrix:
            x = Embedding(input_dim=weights.shape[0],
                          output_dim=weights.shape[1],
                          input_length=input_shape[0],
                          weights=[weights],
                          trainable=is_embedding_trainable)(inp)
            embeddings.append(x)
        x = Concatenate(axis=2)(embeddings)

    if meta_embeddings == 'DME':
        for weights in embedding_matrix:
            x = Embedding(input_dim=weights.shape[0],
                          output_dim=weights.shape[1],
                          input_length=input_shape[0],
                          weights=[weights],
                          trainable=is_embedding_trainable)(inp)
            x = Dense(300)(x)
            embeddings.append(x)
        x = add(embeddings)

    # x = SpatialDropout1D(0.2)(x)

    if model_type == 0:
        # A Structured Self-attentive Sentence Embedding
        # https://arxiv.org/abs/1703.03130
        h = Bidirectional(CuDNNGRU(hidden_dim, kernel_initializer=glorot_uniform(seed=123), return_sequences=True))(x)
        a = Dense(hidden_dim, kernel_initializer=he_uniform(seed=123), activation='tanh')(h)
        a = Dense(8, kernel_initializer=he_uniform(seed=123), activation="softmax")(a)
        m = dot([a, h], axes=(1, 1))
        x = Flatten()(m)
        x = Dense(8 * hidden_dim * 2, kernel_initializer=he_uniform(seed=123), activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(8 * hidden_dim * 2, kernel_initializer=he_uniform(seed=123), activation="relu")(x)
        x = Dropout(dropout_rate)(x)
    if model_type == 1:
        # https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52644
        x = Bidirectional(CuDNNLSTM(hidden_dim, kernel_initializer=glorot_uniform(seed=123), return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(hidden_dim, kernel_initializer=glorot_uniform(seed=123), return_sequences=True))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, max_pool])
        x = Dense(64, kernel_initializer=he_uniform(seed=123), activation="relu")(x)
        x = Dropout(0.1)(x)
    if model_type == 2:
        x = Bidirectional(CuDNNLSTM(hidden_dim, kernel_initializer=glorot_uniform(seed=123), return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(hidden_dim, kernel_initializer=glorot_uniform(seed=123), return_sequences=True))(x)
        x = Attention(MAX_SEQUENCE_LENGTH)(x)
        x = Dense(64, kernel_initializer=he_uniform(seed=123), activation="relu")(x)
        x = Dropout(0.1)(x)
    if model_type == 3:
        # Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm
        # https://arxiv.org/abs/1708.00524
        # https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52563
        lstm_1 = Bidirectional(CuDNNLSTM(hidden_dim, kernel_initializer=glorot_uniform(seed=123), return_sequences=True))(x)
        lstm_2 = Bidirectional(CuDNNLSTM(hidden_dim, kernel_initializer=glorot_uniform(seed=123), return_sequences=True))(lstm_1)
        x = concatenate([lstm_1, lstm_2])
        x = Attention(MAX_SEQUENCE_LENGTH)(x)
    if model_type == 4:
        # Supervised Learning of Universal Sentence Representations from Natural Language Inference Data
        # https://arxiv.org/abs/1705.02364
        x = Bidirectional(CuDNNLSTM(hidden_dim, kernel_initializer=glorot_uniform(seed=123), return_sequences=True))(x)
        x = GlobalMaxPooling1D()(x)

    x = Dense(1, kernel_initializer=he_uniform(seed=123), activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    return model


def model_lstm_atten(embedding_matrix):

    inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = Embedding(input_dim=embedding_matrix.shape[0],
                  output_dim=embedding_matrix.shape[1],
                  weights=[embedding_matrix],
                  trainable=False)(inp)

    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
    y = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)

    atten_1 = Attention(MAX_SEQUENCE_LENGTH)(x)  # skip connect
    atten_2 = Attention(MAX_SEQUENCE_LENGTH)(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)

    conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
    conc = Dense(16, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    return model


def get_best_threshold(y_pred_val,
                       y_val):
    threshold_dict = {}
    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        threshold_dict[thresh] = f1_score(
            y_val, (y_pred_val > thresh).astype(int)
        )

    best_threshold = max(threshold_dict, key=threshold_dict.get)
    print("best threshold: {}".format(best_threshold))
    print("best f1 score: {}".format(threshold_dict[best_threshold]))
    return best_threshold


def fit_predict(X_train,
                X_val,
                y_train,
                y_val,
                X_test,
                model,
                epochs=5,
                lr=0.001,
                batch_size=1024):
    with timer('fitting'):
        model_checkpoint = ModelCheckpoint(
            'best.h5',
            save_best_only=True,
            save_weights_only=True
        )

        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(lr=lr, clipvalue=0.5)
        )
        # model.summary()

        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=6,
            batch_size=batch_size,
            callbacks=[model_checkpoint],
            verbose=1
        )

    best_epoch_index = np.array(hist.history['val_loss']).argmin()
    best_val_loss = np.array(hist.history['val_loss']).min()
    print("best val_loss: {}".format(best_val_loss))
    print("best epoch: {}".format(best_epoch_index + 1))
    model.load_weights('best.h5')

    y_pred_val = model.predict(X_val, batch_size=2048)[:, 0]

    with timer('predicting'):
        y_pred = model.predict(X_test, batch_size=2048)[:, 0]

    get_best_threshold(y_pred_val, y_val)
    return y_pred, y_pred_val


def main():
    with timer('load data'):
        train_df = pd.read_csv("../input/train.csv")
        test_df = pd.read_csv("../input/test.csv")
        X = train_df["question_text"].fillna("_na_").values
        y = train_df["target"].values
        X_test = test_df["question_text"].fillna("_na_").values
        qid = test_df["qid"]

    with timer('preprocess'):
        tokenizer = Tokenizer(num_words=MAX_FEATURES)
        tokenizer.fit_on_texts(
            train_df.question_text.tolist() + test_df.question_text.tolist()
        )
        X = tokenizer.texts_to_sequences(X)
        X_test = tokenizer.texts_to_sequences(X_test)
        word_index = tokenizer.word_index

        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
        X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

    glove_embedding = load_embedding_matrix(
        word_index=word_index,
        embedding_path=GLOVE_PATH
    )

    paragram_embedding = load_embedding_matrix(
        word_index=word_index,
        embedding_path=PARAGRAM_PATH
    )

    embedding_matrix = np.mean(
        [glove_embedding, paragram_embedding],
        axis=0
    )

    y_pred_test = []
    y_pred_val = []

    skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=2018)
    skf_y = np.zeros((len(y),))

    with timer("fit and pred"):
        for train_index, val_index in skf.split(X, y):
            X_train = X[train_index]
            X_val = X[val_index]
            y_train = y[train_index]
            y_val = y[val_index]

            lstm = model_lstm_atten(embedding_matrix)

            pred_test, pred_val = fit_predict(
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
                X_test=X_test,
                epochs=6,
                model=lstm,
                lr=0.001,
                batch_size=512
            )

            skf_y[val_index] = pred_val
            y_pred_test.append(pred_test)

    y_pred_test = np.array(y_pred_test).mean(axis=0)

    print("ALL ensemble")
    threshold = get_best_threshold(skf_y, y)
    y_pred = (y_pred_test > threshold).astype(np.int)

    submit_df = pd.DataFrame({"qid": qid, "prediction": y_pred})
    submit_df.to_csv(
        "submission.csv",
        index=False
    )


if __name__ == '__main__':
    main()
