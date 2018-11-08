from contextlib import contextmanager
import gc
import os
import time
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import CuDNNGRU
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GlobalMaxPool1D
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.layers import MaxPooling1D
from keras.layers import SpatialDropout1D
from keras.layers import Reshape
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tqdm import tqdm


MAX_FEATURES = 50000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300
GLOVE_PATH = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
FAST_TEXT_PATH = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def load_embedding_matrix(word_index,
                          embedding_path=None):
    with timer('load embeddings'):
        embeddings_index = {}
        f = open(embedding_path)
        for line in tqdm(f):
            values = line.split(" ")
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))

    with timer('calculate embedding matrix'):
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


def cnn_model(filters=64,
              filter_sizes=[1, 2, 3, 4],
              dropout_rate=0.1,
              input_shape=None,
              is_embedding_trainable=False,
              embedding_matrix=None):

    maxlen = input_shape[0]
    inp = Input(shape=(maxlen,))
    x = Embedding(input_dim=embedding_matrix.shape[0],
                  output_dim=embedding_matrix.shape[1],
                  input_length=maxlen,
                  weights=[embedding_matrix],
                  trainable=is_embedding_trainable)(inp)
    x = SpatialDropout1D(0.4)(x)
    x = Reshape((maxlen, EMBEDDING_DIM, 1))(x)

    conv_0 = Conv2D(filters, kernel_size=(filter_sizes[0], EMBEDDING_DIM),
                    kernel_initializer='normal', activation='elu')(x)
    conv_1 = Conv2D(filters, kernel_size=(filter_sizes[1], EMBEDDING_DIM),
                    kernel_initializer='normal', activation='elu')(x)
    conv_2 = Conv2D(filters, kernel_size=(filter_sizes[2], EMBEDDING_DIM),
                    kernel_initializer='normal', activation='elu')(x)
    conv_3 = Conv2D(filters, kernel_size=(filter_sizes[3], EMBEDDING_DIM),
                    kernel_initializer='normal', activation='elu')(x)

    maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1))(conv_3)

    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
    z = Flatten()(z)
    z = Dropout(dropout_rate)(z)

    outp = Dense(1, activation="sigmoid")(z)
    model = Model(inputs=inp, outputs=outp)
    return model


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
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation="relu")(x)
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
                batch_size=256):
    with timer('fitting'):
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        model_checkpoint = ModelCheckpoint(
            'best.h5',
            save_best_only=True,
            save_weights_only=True
        )
        class_weights = class_weight.compute_class_weight(
            'balanced',
            np.unique(y_train),
            y_train
        )
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam'
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=1000,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=[early_stopping, model_checkpoint]
        )

    model.load_weights('best.h5')
    y_pred_val = model.predict(X_val, batch_size=2048)[:, 0]

    with timer('predicting'):
        y_pred = model.predict(X_test, batch_size=2048)[:, 0]

    get_best_threshold(y_pred_val, y_val)
    return y_pred, y_pred_val


def main():
    with timer('load data'):
        train_df = pd.read_csv("../input/train.csv")
        test_df = pd.read_csv('../input/test.csv')
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

    gru_glove = bigru_model(
        hidden_dim=64,
        dropout_rate=0.1,
        input_shape=X_train.shape[1:],
        is_embedding_trainable=False,
        embedding_matrix=glove_embedding
    )

    gru_glove_pred_test, gru_glove_pred_val = fit_predict(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        X_test=X_test,
        model=gru_glove,
        batch_size=256
    )

    cnn_glove = cnn_model(
        input_shape=X_train.shape[1:],
        is_embedding_trainable=False,
        embedding_matrix=glove_embedding
    )

    cnn_glove_pred_test, cnn_glove_pred_val = fit_predict(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        X_test=X_test,
        model=cnn_glove,
        batch_size=256
    )

    del glove_embedding
    gc.collect()

    fast_text_embedding = load_embedding_matrix(
        word_index=word_index,
        embedding_path=FAST_TEXT_PATH
    )

    gru_fast_text = bigru_model(
        hidden_dim=64,
        dropout_rate=0.1,
        input_shape=X_train.shape[1:],
        is_embedding_trainable=False,
        embedding_matrix=fast_text_embedding
    )

    gru_fast_text_pred_test, gru_fast_text_pred_val = fit_predict(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        X_test=X_test,
        model=gru_fast_text,
        batch_size=256
    )

    cnn_fast_text = cnn_model(
        input_shape=X_train.shape[1:],
        is_embedding_trainable=False,
        embedding_matrix=fast_text_embedding
    )

    cnn_fast_text_pred_test, cnn_fast_text_pred_val = fit_predict(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        X_test=X_test,
        model=cnn_fast_text,
        batch_size=256
    )

    del fast_text_embedding
    gc.collect()

    gru_pred_val = (gru_glove_pred_val + gru_fast_text_pred_val) / 2
    cnn_pred_val = (cnn_glove_pred_val + cnn_fast_text_pred_val) / 2

    gru_pred_test = (gru_glove_pred_test + gru_fast_text_pred_test) / 2
    cnn_pred_test = (cnn_glove_pred_test + cnn_fast_text_pred_test) / 2

    y_pred_val = 0.6 * gru_pred_val + 0.4 * cnn_pred_val
    y_pred_test = 0.6 * gru_pred_test + 0.4 * cnn_pred_test

    threshold = get_best_threshold(y_pred_val, y_val)
    y_pred = (np.array(y_pred_test) > threshold).astype(np.int)

    submit_df = pd.DataFrame({"qid": qid, "prediction": y_pred})
    submit_df.to_csv(
        "submission.csv",
        index=False
    )


if __name__ == '__main__':
    main()
