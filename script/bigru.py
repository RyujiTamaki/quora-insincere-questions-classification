from contextlib import contextmanager
import os
import time
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import CuDNNGRU
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tqdm import tqdm


TOP_K = 20000
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 300


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class F1Evaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            y_pred = (y_pred > 0.5).astype(int)
            score = f1_score(self.y_val, y_pred)
            print("\n F1 Score - epoch: %d - score: %.6f \n" % (epoch+1, score))


def bigru_model(hidden_dim,
                dropout_rate,
                input_shape,
                is_embedding_trainable=False,
                embedding_matrix=None):
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0],
                        output_dim=embedding_matrix.shape[1],
                        input_length=input_shape[0],
                        weights=[embedding_matrix],
                        trainable=is_embedding_trainable))

    model.add(Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True)))
    model.add(Bidirectional(CuDNNGRU(hidden_dim)))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    return model


def main():
    with timer('load data'):
        train_df = pd.read_csv("../input/train.csv")
        test_df = pd.read_csv('../input/test.csv')

    with timer('load embeddings'):
        embeddings_index = {}
        f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
        for line in tqdm(f):
            values = line.split(" ")
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))

    with timer('preprocess'):
        tokenizer = Tokenizer(num_words=TOP_K)
        tokenizer.fit_on_texts(np.hstack((
            train_df.question_text.values, test_df.question_text.values))
        )
        X_train = tokenizer.texts_to_sequences(train_df.question_text.values)
        X_test = tokenizer.texts_to_sequences(test_df.question_text.values)
        word_index = tokenizer.word_index

        max_length = len(max(X_train, key=len))
        if max_length > MAX_SEQUENCE_LENGTH:
            max_length = MAX_SEQUENCE_LENGTH

        X_train = pad_sequences(X_train, maxlen=max_length)
        X_test = pad_sequences(X_test, maxlen=max_length)

    with timer('calculate embedding matrix'):
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    with timer('fitting'):
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            train_df.target.values,
            test_size=0.1,
            random_state=39
        )
        class_weights = class_weight.compute_class_weight(
            'balanced',
            np.unique(train_df.target.values),
            train_df.target.values
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        f1_Score = F1Evaluation(validation_data=(X_val, y_val), interval=1)
        model_checkpoint = ModelCheckpoint(
            'best_gru.h5',
            save_best_only=True,
            save_weights_only=True
        )
        model = bigru_model(
            hidden_dim=64,
            dropout_rate=0.2,
            input_shape=X_train.shape[1:],
            is_embedding_trainable=False,
            embedding_matrix=embedding_matrix
        )

        model.compile(
            loss='binary_crossentropy',
            optimizer='adam'
        )

        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=1000,
            batch_size=128,
            class_weight=class_weights,
            callbacks=[early_stopping, model_checkpoint, f1_Score]
        )

    with timer('predicting'):
        y_pred = model.predict(X_test, batch_size=1024)[:, 0]
        y_pred = (np.array(y_pred) > 0.5).astype(np.int)

        submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_pred})
        submit_df.to_csv(
            "submission.csv",
            index=False
        )


if __name__ == '__main__':
    main()
