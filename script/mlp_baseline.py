import os; os.environ['OMP_NUM_THREADS'] = '1'
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
from typing import List, Dict

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.utils import class_weight


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)


def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')


def build_mlp_model(input_dim):
    model_in = tf.keras.Input(shape=(input_dim,), dtype='float32', sparse=True)
    out = tf.keras.layers.Dense(128, input_shape=(input_dim, ), activation='relu')(model_in)
    out = tf.keras.layers.Dense(64, activation='relu')(out)
    out = tf.keras.layers.Dense(32, activation='relu')(out)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(out)
    model = tf.keras.Model(model_in, out)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=3e-3))
    return model


def main():
    vectorizer = make_union(
        on_field('question_text', TfidfVectorizer(
                    max_features=20000,
                    analyzer='word',
                    token_pattern='\w+',
                    stop_words='english',
                    ngram_range=(1, 2)
                )),
        on_field('question_text', TfidfVectorizer(
                    max_features=20000,
                    analyzer='char',
                    token_pattern='\w+',
                    ngram_range=(3, 3)
                )),
        n_jobs=4
    )

    with timer('load data'):
        train = pd.read_csv("../input/train.csv")
        test = pd.read_csv('../input/test.csv')

    with timer('process train'):
        cv = KFold(n_splits=20, shuffle=True, random_state=42)
        train_ids, valid_ids = next(cv.split(train))
        train, valid = train.iloc[train_ids], train.iloc[valid_ids]
        y_train = train.target.values
        y_valid = valid.target.values
        X_train = vectorizer.fit_transform(train).astype(np.float32)
        print(f'X_train: {X_train.shape} of {X_train.dtype}')
        class_weights = class_weight.compute_class_weight(
            'balanced',
            np.unique(train.target.values),
            train.target.values
        )
        del train

    with timer('process valid'):
        X_valid = vectorizer.transform(valid).astype(np.float32)

    with timer('process test'):
        X_test = vectorizer.transform(test).astype(np.float32)

    with timer('fitting'):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint('mlp.h5', save_best_only=True, save_weights_only=True)
        model = build_mlp_model(input_dim=X_train.shape[1])
        model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_data=[X_valid, y_valid],
            callbacks=[early_stopping, model_checkpoint],
            class_weight=class_weights,
            verbose=True
        )

        json_string = model.to_json()
        del model
        model = tf.keras.models.model_from_json(json_string)
        model.load_weights('mlp.h5')
        y_pred_val = model.predict(X_valid, batch_size=8192)[:, 0]
        y_pred_val_bin = (np.array(y_pred_val) > 0.5).astype(np.int)
        val_score = f1_score(y_pred=y_pred_bin, y_true=valid.target.values)
        print(f"f1 score: {val_score}")

    with timer('predicting'):
        y_pred_test = model.predict(X_test, batch_size=8192)[:, 0]
        y_pred_test_bin = (np.array(y_pred_test) > 0.5).astype(np.int)
        submit_df = pd.DataFrame({
            "qid": test["qid"], "prediction": y_pred_test_bin
        })
        submit_df.to_csv(f"f1_{val_score}.csv", index=False)


if __name__ == '__main__':
    main()
