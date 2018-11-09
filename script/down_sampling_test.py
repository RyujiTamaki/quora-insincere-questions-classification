from contextlib import contextmanager
import gc
import os
import time
import numpy as np
import pandas as pd
from keras import backend as K
from keras import regularizers
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional
from keras.layers import Concatenate
from keras.layers import CuDNNGRU
from keras.layers import dot
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GlobalMaxPool1D
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Reshape
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


def load_embedding_matrix(word_index,
                          embedding_path=None):
    with timer('load embeddings'):
        embeddings_index = {}
        f = open(embedding_path)
        for line in f:
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


def preprocessing_text(question_text, tokenizer):
    tokenized_text = tokenizer.texts_to_sequences(question_text)
    return pad_sequences(tokenized_text, maxlen=MAX_SEQUENCE_LENGTH)


def bigru_attn_model(hidden_dim,
                     dropout_rate,
                     input_shape,
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
    # (MAX_SEQUENCE_LENGTH, hidden_dim)
    a = Dense(hidden_dim, activation='tanh')(h)
    # (MAX_SEQUENCE_LENGTH, 8)
    a = Dense(8, activation="sigmoid")(a)
    # (8, hidden * 2)
    m = dot([a, h], axes=(1, 1))
    # (8 * hidden * 2)
    x = Flatten()(m)
    x = Dense(8 * hidden_dim * 2, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(8 * hidden_dim * 2, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
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
            verbose=1
        )

    model.load_weights('best.h5')
    y_pred_val = model.predict(X_val, batch_size=2048)[:, 0]

    with timer('predicting'):
        y_pred = model.predict(X_test, batch_size=2048)[:, 0]

    get_best_threshold(y_pred_val, y_val)
    return y_pred, y_pred_val


def down_sampling(df, random_state):
    positive_data = df[df.target == 1]
    positive_ratio = float(len(positive_data)) / len(df)
    negative_data = df[df.target == 0].sample(
        frac=positive_ratio / (1 - positive_ratio), random_state=random_state
    )
    return positive_data.index.union(negative_data.index).sort_values()


def main():
    with timer('load data'):
        train_df = pd.read_csv("../input/train.csv")
        test_df = pd.read_csv('../input/test.csv')
        qid = test_df["qid"]

    with timer('tokenizer.fit_on_texts'):
        tokenizer = Tokenizer(num_words=MAX_FEATURES)
        tokenizer.fit_on_texts(np.hstack((
            train_df.question_text.values, test_df.question_text.values))
        )
        word_index = tokenizer.word_index

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        random_state=39
    )

    with timer('preprocessing_text'):
        X_val = preprocessing_text(
            question_text=val_df.question_text.values,
            tokenizer=tokenizer
        )
        X_test = preprocessing_text(
            question_text=test_df.question_text.values,
            tokenizer=tokenizer
        )

    y_val = val_df.target.values

    glove_embedding = load_embedding_matrix(
        word_index=word_index,
        embedding_path=GLOVE_PATH
    )

    attn_glove_pred_test = []
    attn_glove_pred_val = []

    for i in range(10):
        sample_df = down_sampling(train_df, i)
        X_train = preprocessing_text(
            question_text=sample_df.question_text.values,
            tokenizer=tokenizer
        )
        y_test = sample_df.target.values

        attn_glove = bigru_attn_model(
            hidden_dim=64,
            dropout_rate=0.5,
            input_shape=X_train.shape[1:],
            is_embedding_trainable=False,
            embedding_matrix=glove_embedding
        )

        pred_test, pred_val = fit_predict(
            X_train=X_train,
            X_val=X_val,
            y_train=y_test,
            y_val=y_val,
            X_test=X_test,
            model=attn_glove,
            batch_size=1024
        )

        attn_glove_pred_test.append(pred_test)
        attn_glove_pred_val.append(pred_val)

        del sample_df, attn_glove
        gc.collect()

    del glove_embedding
    gc.collect()

    threshold = get_best_threshold(attn_glove_pred_val, y_val)
    y_pred = (np.array(attn_glove_pred_test) > threshold).astype(np.int)

    submit_df = pd.DataFrame({"qid": qid, "prediction": y_pred})
    submit_df.to_csv(
        "submission.csv",
        index=False
    )


if __name__ == '__main__':
    main()
