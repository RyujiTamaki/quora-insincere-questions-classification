from contextlib import contextmanager
import gc
import os
import time
import numpy as np
import pandas as pd
import random as rn
from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import AveragePooling1D
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.layers import Concatenate
from keras.layers import CuDNNGRU
from keras.layers import dot
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import PReLU
from keras.layers import Reshape
from keras.layers import SpatialDropout1D
from keras.models import Model
from keras.layers import Conv1D, MaxPool1D, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf


# os.environ['PYTHONHASHSEED'] = '0'
# np.random.seed(42)
# rn.seed(12345)

MAX_FEATURES = 50000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300
GLOVE_PATH = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
FAST_TEXT_PATH = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
PARAGRAM_PATH = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def load_embedding_matrix(word_index,
                          embedding_path=None):
    with timer('load embeddings'):
        embeddings_index = {}
        if embedding_path == PARAGRAM_PATH:
            f = open(embedding_path, encoding="utf8", errors='ignore')
        else:
            f = open(embedding_path)
        for line in f:
            values = line.split(" ")
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))

        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


def bigru_attn_model(hidden_dim,
                     dropout_rate,
                     input_shape,
                     pool_type='attn',
                     is_embedding_trainable=False,
                     embedding_matrix=None):

    inp = Input(shape=(input_shape[0],))
    # (MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
    x = Embedding(input_dim=embedding_matrix.shape[0],
                  output_dim=embedding_matrix.shape[1],
                  input_length=input_shape[0],
                  weights=[embedding_matrix],
                  trainable=is_embedding_trainable)(inp)
    # (MAX_SEQUENCE_LENGTH, hidden_dim * 2)
    h = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))(x)

    if pool_type == 'attn':
        # (MAX_SEQUENCE_LENGTH, hidden_dim)
        a = Dense(hidden_dim, activation='tanh')(h)
        # (MAX_SEQUENCE_LENGTH, 8)
        a = Dense(8, activation="softmax")(a)
        # (8, hidden * 2)
        m = dot([a, h], axes=(1, 1))
        # (8 * hidden * 2)
        x = Flatten()(m)
    if pool_type == 'max':
        x = GlobalMaxPooling1D()(h)

    x = Dense(1024, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(1024, activation="relu")(x)
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

        model.summary()

        val_loss = []
        for i in range(5):
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
                batch_size=batch_size * (1 + i),
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
        tokenizer.fit_on_texts(np.hstack((
            train_df.question_text.values, test_df.question_text.values))
        )
        X_train = tokenizer.texts_to_sequences(train_df.question_text.values)
        X_test = tokenizer.texts_to_sequences(test_df.question_text.values)
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

    embeddings = [glove_embedding, glove_embedding,
                  fast_text_embedding, paragram_embedding]

    gru_attn_pred_test = []
    gru_attn_pred_val = []

    for embedding_matrix in embeddings:
        for _ in range(2):
            gru_attn = bigru_attn_model(
                hidden_dim=64,
                dropout_rate=0.5,
                input_shape=X_train.shape[1:],
                pool_type='attn',
                is_embedding_trainable=False,
                embedding_matrix=embedding_matrix
            )

            pred_test, pred_val = fit_predict(
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
                X_test=X_test,
                model=gru_attn,
                lr=0.001,
                batch_size=1024
            )

            gru_attn_pred_test.append(pred_test)
            gru_attn_pred_val.append(pred_val)

            del gru_attn
            gc.collect()

    gru_attn_pred_val = np.array(gru_attn_pred_val).mean(axis=0)
    gru_attn_pred_test = np.array(gru_attn_pred_test).mean(axis=0)

    print("ALL ensemble")
    threshold = get_best_threshold(gru_attn_pred_val, y_val)
    y_pred = (np.array(gru_attn_pred_test) > threshold).astype(np.int)

    submit_df = pd.DataFrame({"qid": qid, "prediction": y_pred})
    submit_df.to_csv(
        "submission.csv",
        index=False
    )


if __name__ == '__main__':
    main()