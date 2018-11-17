from contextlib import contextmanager
import gc
import os
import time
import numpy as np
import pandas as pd
import joblib
from keras import backend as K
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.engine.topology import Layer
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import CuDNNGRU
from keras.layers import CuDNNLSTM
from keras.layers import dot
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GlobalMaxPool1D
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import PReLU
from keras.layers import Reshape
from keras.layers import *
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


MAX_FEATURES = 50000
MAX_SEQUENCE_LENGTH = 100
GLOVE_PATH = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
FAST_TEXT_PATH = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    TIME_STEPS = inputs.shape[1].value
    SINGLE_ATTENTION_VECTOR = False

    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    # this line is not useful. It's just to know which dimension is what.
    a = Reshape((input_dim, TIME_STEPS))(a)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


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


def bigru_model(hidden_dim,
                dropout_rate,
                input_shape,
                is_embedding_trainable=False,
                embedding_matrix=None):

    inp = Input(shape=(input_shape[0],))
    x = Embedding(input_dim=embedding_matrix.shape[0],
                  output_dim=embedding_matrix.shape[1],
                  input_length=input_shape[0],
                  weights=[embedding_matrix],
                  trainable=is_embedding_trainable)(inp)

    x = SpatialDropout1D(0.2)(x)

    x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)
    x = Attention(MAX_SEQUENCE_LENGTH)(x)
    # x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    return model


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
            x = Dense(300)(x)
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
        x = Attention(MAX_SEQUENCE_LENGTH)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.1)(x)
    if model_type == 2:
        x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, max_pool])
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.1)(x)

    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
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
                epochs=3,
                lr=0.001,
                batch_size=1024):
    with timer('fitting'):
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        class_weights = class_weight.compute_class_weight(
            'balanced',
            np.unique(y_train),
            y_train
        )
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(lr=lr, clipvalue=0.5)
        )

        model.summary()

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
                # batch_size=2**(9 + i),
                batch_size=batch_size * (i + 1),
                # batch_size=512,
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
        test_df = pd.read_csv('../input/test.csv')
        X_train = joblib.load('../input/X_train.joblib')
        y_train = joblib.load('../input/y_train.joblib')
        X_test = joblib.load('../input/X_test.joblib')
        glove_embedding = joblib.load('../input/glove_embedding.joblib')
        fast_text_embedding = joblib.load('../input/fast_text_embedding.joblib')
        paragram_embedding = joblib.load('../input/paragram_embedding.joblib')
        word2vec_embedding = joblib.load('../input/word2vec_embedding.joblib')
        test_df = pd.read_csv('../input/test.csv')
        qid = test_df["qid"]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=39
    )

    embedding_matrix = [
        glove_embedding, fast_text_embedding, paragram_embedding, word2vec_embedding
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

    y_pred_test = np.array(y_pred_test).mean(axis=0)
    y_pred_val = np.array(y_pred_val).mean(axis=0)

    print("ALL ensemble")
    threshold = get_best_threshold(y_pred_val, y_val)
    y_pred = (np.array(y_pred_test) > threshold).astype(np.int)


if __name__ == '__main__':
    main()
