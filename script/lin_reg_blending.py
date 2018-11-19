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
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf


# os.environ['PYTHONHASHSEED'] = '0'
# np.random.seed(42)
# rn.seed(12345)

MAX_FEATURES = 95000
MAX_SEQUENCE_LENGTH = 70
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
        return input_shape[0],  self.features_dim


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
                    embedding_vector = None
            else:
                embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            if embedding_vector is None:
                embedding_matrix[i] = np.random.uniform(-0.01, 0.01, EMBEDDING_DIM)

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
            # x = Dense(300)(x)
            embeddings.append(x)
        x = add(embeddings)

    if model_type == 0:
        h = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))(x)
        a = Dense(hidden_dim, activation='tanh')(h)
        a = Dense(8, activation="softmax")(a)
        m = dot([a, h], axes=(1, 1))
        x = Flatten()(m)
        x = Dense(8 * hidden_dim * 2, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(8 * hidden_dim * 2, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
    if model_type == 1:
        x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, max_pool])
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.1)(x)
    if model_type == 2:
        x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)
        x = Attention(MAX_SEQUENCE_LENGTH)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.1)(x)

    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    return model


def cnn_model(filters=32,
              kernel_sizes=[1, 2, 3, 4],
              dropout_rate=0.1,
              input_shape=None,
              is_embedding_trainable=False,
              meta_embeddings='concat',
              embedding_matrix=None):

    maxlen = input_shape[0]
    inp = Input(shape=(maxlen,))
    embeddings = []
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
    # x = SpatialDropout1D(0.4)(x)

    convs = []
    for kernel_size in kernel_sizes:
        conv = Conv1D(filters, kernel_size=(kernel_size),
                      kernel_initializer='he_normal', activation='tanh')(x)
        maxpool = MaxPool1D(pool_size=(maxlen - kernel_size + 1))(conv)
        convs.append(maxpool)

    x = Concatenate(axis=1)(convs)
    x = Flatten()(x)
    # x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    outp = Dense(1, activation="sigmoid")(x)
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
        class_weights = class_weight.compute_class_weight(
            'balanced',
            np.unique(y_train),
            y_train
        )

        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(lr=lr)
        )
        # model.summary()
        val_loss = []
        for i in range(epochs):
            model_checkpoint = ModelCheckpoint(
                str(i) + '_weight.h5',
                save_best_only=True,
                save_weights_only=True
            )

            hist = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=1,
                # batch_size=2**(11 + i),
                batch_size=batch_size,  # *(i + 1),
                class_weight=class_weights,
                callbacks=[model_checkpoint],
                verbose=2
            )

            val_loss.extend(hist.history['val_loss'])

    best_epoch_index = np.array(val_loss).argmin()
    print("best epoch: {}".format(best_epoch_index + 1))
    model.load_weights(str(best_epoch_index) + '_weight.h5')

    y_pred_val = model.predict(X_val, batch_size=2048)[:, 0]

    with timer('predicting'):
        y_pred = model.predict(X_test, batch_size=2048)[:, 0]

    get_best_threshold(y_pred_val, y_val)
    return y_pred, y_pred_val


def main():
    with timer('load data'):
        train_df = pd.read_csv("../input/train.csv")
        test_df = pd.read_csv("../input/test.csv")
        X_train = train_df["question_text"].fillna("_na_").values
        y_train = train_df["target"].values
        X_test = test_df["question_text"].fillna("_na_").values
        qid = test_df["qid"]

    with timer('preprocess'):
        tokenizer = Tokenizer(num_words=MAX_FEATURES)
        tokenizer.fit_on_texts(
            train_df.question_text.tolist() + test_df.question_text.tolist()
        )
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)
        word_index = tokenizer.word_index

        X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
        X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=39
    )

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
    '''
    word2vec_embedding = load_embedding_matrix(
        word_index=word_index,
        embedding_path=WORD2VEC_PATH
    )
    '''

    embedding_matrix = [
        glove_embedding, fast_text_embedding, paragram_embedding  # , word2vec_embedding
    ]

    y_pred_test = []
    y_pred_val = []

    for i in range(6):
        gru = build_gru(
            hidden_dim=40,
            dropout_rate=0.1,
            input_shape=X_train.shape[1:],
            model_type=i % 3,
            is_embedding_trainable=False,
            meta_embeddings='concat',
            embedding_matrix=embedding_matrix
        )

        pred_test, pred_val = fit_predict(
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            X_test=X_test,
            epochs=3,
            model=gru,
            lr=0.001,
            batch_size=1024
        )

        y_pred_test.append(pred_test)
        y_pred_val.append(pred_val)

    print('CNN fitting')
    for _ in range(2):
        cnn = cnn_model(
            filters=40,
            kernel_sizes=[2, 3, 4, 5],
            input_shape=X_train.shape[1:],
            is_embedding_trainable=True,
            meta_embeddings='concat',
            embedding_matrix=embedding_matrix
        )

        pred_test, pred_val = fit_predict(
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            X_test=X_test,
            model=cnn,
            lr=0.001,
            epochs=1,
            batch_size=512
        )

        y_pred_test.append(pred_test)
        y_pred_val.append(pred_val)

    # y_pred_test = np.array(y_pred_test).mean(axis=0)
    # y_pred_val = np.array(y_pred_val).mean(axis=0)
    print("mean ensemble")
    threshold = get_best_threshold(np.array(y_pred_val).mean(axis=0), y_val)
    y_pred = (np.array(np.array(y_pred_test).mean(axis=0)) > threshold).astype(np.int)

    X = np.array(y_pred_val)
    reg = LinearRegression().fit(X.T, y_val)
    print(reg.score(X.T, y_val), reg.coef_)

    y_pred_val = np.sum(
        [y_pred_val[i] * reg.coef_[i] for i in range(len(y_pred_val))], axis=0)
    y_pred_test = np.sum(
        [y_pred_test[i] * reg.coef_[i] for i in range(len(y_pred_test))], axis=0)

    print("LinearRegression ensemble")
    threshold = get_best_threshold(y_pred_val, y_val)
    y_pred = (np.array(y_pred_test) > threshold).astype(np.int)

    submit_df = pd.DataFrame({"qid": qid, "prediction": y_pred})
    submit_df.to_csv(
        "submission.csv",
        index=False
    )


if __name__ == '__main__':
    main()
