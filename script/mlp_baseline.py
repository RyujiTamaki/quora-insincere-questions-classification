from contextlib import contextmanager
import gc
import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict
from operator import itemgetter


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
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)


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
                batch_size=32):
    with timer('fitting'):
        early_stopping = EarlyStopping(monitor='val_loss', patience=1)
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


def build_mlp_model(input_dim):
    model_in = Input(shape=(input_dim,), dtype='float32', sparse=True)
    out = Dense(128, activation='relu')(model_in)
    out = Dense(64, activation='relu')(out)
    out = Dense(32, activation='relu')(out)
    out = Dense(1, activation='sigmoid')(out)
    model = Model(model_in, out)
    return model


def main():
    vectorizer = make_union(
        on_field('question_text', TfidfVectorizer(
                    max_features=100000,
                    analyzer='word',
                    token_pattern='\w+',
                    stop_words='english',
                    ngram_range=(1, 2)
                )),
        on_field('question_text', TfidfVectorizer(
                    max_features=100000,
                    analyzer='char_wb',
                    token_pattern='\w+',
                    ngram_range=(3, 3)
                )),
        n_jobs=4
    )

    with timer('load data'):
        train_df = pd.read_csv("../input/train.csv")
        test_df = pd.read_csv('../input/test.csv')
        all_text = pd.concat([train_df, test_df])
        y_train = train_df["target"].values
        qid = test_df["qid"]

    with timer('process train'):
        vectorizer.fit_transform(all_text)
        X_train = vectorizer.transform(train_df)
        X_test = vectorizer.transform(test_df)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=39
    )

    mlp_model = build_mlp_model(input_dim=X_train.shape[1])

    mlp_pred_test, mlp_pred_val = fit_predict(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        X_test=X_test,
        model=mlp_model,
        batch_size=2048
    )

    threshold = get_best_threshold(mlp_pred_val, y_val)
    y_pred = (np.array(mlp_pred_test) > threshold).astype(np.int)

    submit_df = pd.DataFrame({"qid": qid, "prediction": y_pred})
    submit_df.to_csv(
        "submission.csv",
        index=False
    )


if __name__ == '__main__':
    main()
