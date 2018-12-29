from contextlib import contextmanager
import gc
import os
import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.model_selection import train_test_split

import optuna

max_features = 95000
maxlen = 70
embed_size = 300
n_epochs = 6

loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')

glove_embedding = joblib.load('../input/glove_embedding.joblib')
paragram_embedding = joblib.load('../input/paragram_embedding.joblib')
embedding_matrix = np.mean([glove_embedding, paragram_embedding], axis=0)


def seed_torch(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class NeuralNet(nn.Module):
    def __init__(self,
                 dropout_rate=0.1,
                 lstm_hidden_size=40,
                 last_hidden_size=16):
        super(NeuralNet, self).__init__()

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(dropout_rate)
        self.lstm = nn.LSTM(embed_size, lstm_hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(lstm_hidden_size * 2, lstm_hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(lstm_hidden_size * 2, maxlen)
        self.gru_attention = Attention(lstm_hidden_size * 2, maxlen)

        self.linear = nn.Linear(lstm_hidden_size * 8, last_hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(last_hidden_size, 1)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(
            self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)

        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)

        # global average pooling
        avg_pool = torch.mean(h_gru, 1)
        # global max pooling
        max_pool, _ = torch.max(h_gru, 1)

        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out


def create_model(trial):
    lstm_hidden_size = int(trial.suggest_loguniform('lstm_hidden_size', 16, 512))
    last_hidden_size = int(trial.suggest_loguniform('last_hidden_size', 4, 128))
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)

    model = NeuralNet(
        dropout_rate=dropout_rate,
        lstm_hidden_size=lstm_hidden_size,
        last_hidden_size=last_hidden_size
    )
    model.cuda()

    return model


def get_optimizer(trial, model):
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
    return optimizer


def train_model(model, train_loader, optimizer):
    model.train()
    avg_loss = 0.
    for x_batch, y_batch in train_loader:
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)

    return avg_loss


def test_model(model, test_loader):
    model.eval()
    avg_val_loss = 0.
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_pred = model(x_batch).detach()
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(test_loader)

    return avg_val_loss


def objective(trial):
    # dataset
    X_train = joblib.load('../input/X_train.joblib')
    y_train = joblib.load('../input/y_train.joblib')

    X_train, X_test, y_train, y_test = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=39
    )

    x_train_tensor = torch.tensor(X_train, dtype=torch.long).cuda()
    y_train_tensor = torch.tensor(y_train[:, np.newaxis], dtype=torch.float32).cuda()
    x_test_tensor = torch.tensor(X_test, dtype=torch.long).cuda()
    y_test_tensor = torch.tensor(y_test[:, np.newaxis], dtype=torch.float32).cuda()

    train = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    test = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

    batch_size = int(trial.suggest_loguniform('batch_size', 128, 1024))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    # model
    model = create_model(trial)
    optimizer = get_optimizer(trial, model)
    val_loss = []

    for step in range(n_epochs):
        start_time = time.time()
        avg_loss = train_model(model, train_loader, optimizer)
        avg_val_loss = test_model(model, test_loader)
        val_loss.append(avg_val_loss)

        trial.report(avg_val_loss, step)

        if trial.should_prune(step):
            raise optuna.structs.TrialPruned()

        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
            step + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))

    return min(val_loss)


def main():
    seed_torch()
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    print('  User attrs:')
    for key, value in trial.user_attrs.items():
        print('    {}: {}'.format(key, value))


if __name__ == '__main__':
    main()
