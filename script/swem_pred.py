from contextlib import contextmanager
import gc
import os
import time
import numpy as np
import pandas as pd
import joblib
from keras import backend as K
from keras import regularizers
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import AveragePooling1D
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
from keras.layers import concatenate
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
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


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


def swem(dropout_rate,
         input_shape,
         embedding_matrix=None,
         pool_type='max'):

    inp = Input(shape=(input_shape[0],))
    x = Embedding(input_dim=embedding_matrix.shape[0],
                  output_dim=embedding_matrix.shape[1],
                  input_length=input_shape[0],
                  weights=[embedding_matrix],
                  trainable=False)(inp)

    x = SpatialDropout1D(rate=dropout_rate)(x)
    x = Dense(300, activation="relu")(x)

    if pool_type == 'aver':
        x = GlobalAveragePooling1D()(x)
    if pool_type == 'max':
        x = GlobalMaxPooling1D()(x)
    if pool_type == 'concat':
        x_aver = GlobalAveragePooling1D()(x)
        x_max = GlobalMaxPooling1D()(x)
        x = concatenate([x_aver, x_max])
    if pool_type == 'hier':
        x = AveragePooling1D(pool_size=5,
                             strides=None,
                             padding='same')(x)
        x = GlobalMaxPooling1D()(x)

    x = Dense(300, activation="relu")(x)
    x = Dropout(rate=dropout_rate)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.summary()
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
                batch_size=1024):
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
            callbacks=[early_stopping, model_checkpoint],
            verbose=2
        )

    model.load_weights('best.h5')
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
        test_df = pd.read_csv('../input/test.csv')
        qid = test_df["qid"]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=39
    )

    swem_hier = swem(
        dropout_rate=0.1,
        input_shape=X_train.shape[1:],
        embedding_matrix=glove_embedding,
        pool_type='hier'
    )

    swem_hier_pred_test, swem_hier_pred_val = fit_predict(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        X_test=X_test,
        model=swem_hier,
        batch_size=1024
    )

    y_pred_val = (swem_max_pred_val + swem_aver_pred_val +
                  swem_concat_pred_val + swem_hier_pred_val) / 4

    y_pred_test = (swem_max_pred_test + swem_aver_pred_test +
                   swem_concat_pred_test + swem_hier_pred_test) / 4

    del glove_embedding
    gc.collect()

    threshold = get_best_threshold(y_pred_val, y_val)
    y_pred = (np.array(y_pred_test) > threshold).astype(np.int)

    submit_df = pd.DataFrame({"qid": qid, "prediction": y_pred})
    submit_df.to_csv(
        "submission.csv",
        index=False
    )


if __name__ == '__main__':
    main()