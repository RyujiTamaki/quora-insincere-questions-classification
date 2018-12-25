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
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
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


# https://www.kaggle.com/hireme/fun-api-keras-f1-metric-cyclical-learning-rate/code

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


def f1(y_true, y_pred):
    '''
    metric from here
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def load_embedding_matrix(word_index,
                          embedding_path=None):
    with timer('load embeddings'):
        num_words = min(MAX_FEATURES, len(word_index)) + 1
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        embeddings_index = {}

        if embedding_path == WORD2VEC_PATH:
            embeddings_index = KeyedVectors.load_word2vec_format(
                embedding_path, binary=True)
        else:
            f = open(embedding_path, encoding="utf8", errors='ignore')
            for line in f:
                values = line.split(" ")
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()

        for word, i in word_index.items():
            if i >= MAX_FEATURES:
                continue
            if word in embeddings_index:
                embedding_matrix[i] = embeddings_index[word]

    return embedding_matrix


def build_gru(input_shape,
              hidden_dim=40,
              dropout_rate=0.1,
              model_type=0,
              is_embedding_trainable=False,
              meta_embeddings=None,
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

    if meta_embeddings is None:
        x = Embedding(input_dim=embedding_matrix.shape[0],
                      output_dim=embedding_matrix.shape[1],
                      input_length=input_shape[0],
                      weights=[embedding_matrix],
                      trainable=is_embedding_trainable)(inp)

    x = SpatialDropout1D(dropout_rate)(x)

    if model_type == 0:
        # A Structured Self-attentive Sentence Embedding https://arxiv.org/abs/1703.03130
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
        x = Bidirectional(CuDNNLSTM(hidden_dim, return_sequences=True))(x)
        y = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))(x)

        atten_1 = Attention(MAX_SEQUENCE_LENGTH)(x)  # skip connect
        atten_2 = Attention(MAX_SEQUENCE_LENGTH)(y)
        avg_pool = GlobalAveragePooling1D()(y)
        max_pool = GlobalMaxPooling1D()(y)

        x = concatenate([atten_1, atten_2, avg_pool, max_pool])
        x = Dense(16, activation="relu")(x)
        x = Dropout(dropout_rate)(x)

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
                epochs=5,
                lr=0.001,
                batch_size=512):
    with timer('fitting'):
        clr = CyclicLR(
            base_lr=0.001,
            max_lr=0.002,
            step_size=300.,
            mode='exp_range',
            gamma=0.99994
        )
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(lr=lr, clipvalue=0.5)
        )
        # model.summary()
        model_checkpoint = ModelCheckpoint(
            'best.h5',
            save_best_only=True,
            save_weights_only=True
        )

        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[model_checkpoint, clr],
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

    '''
    embedding_matrix = [
        glove_embedding, paragram_embedding
    ]
    '''

    embedding_matrix = np.mean([glove_embedding, paragram_embedding], axis = 0)

    y_pred_test = []
    y_pred_val = []

    skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=123)
    skf_y = np.zeros((len(y),))

    with timer("fit and pred"):
        for train_index, val_index in skf.split(X, y):
            X_train = X[train_index]
            X_val = X[val_index]
            y_train = y[train_index]
            y_val = y[val_index]

            gru = build_gru(
                hidden_dim=40,
                dropout_rate=0.1,
                input_shape=X_train.shape[1:],
                model_type=1,
                is_embedding_trainable=False,
                meta_embeddings=None,
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
