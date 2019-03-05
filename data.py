from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np


def split_data():
    df_train = pd.read_csv('data/train.csv')
    X = df_train.values[:, 2:]
    y = df_train.values[:, 1]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
    idx = list(sss.split(X, y))
    X_tr, y_tr = X[idx[0][0]], y[idx[0][0]]
    X_dv, y_dv = X[idx[0][1]], y[idx[0][1]]
    idx = list(sss.split(X_tr, y_tr))
    X_te, y_te = X_tr[idx[0][1]], y_tr[idx[0][1]]
    X_tr, y_tr = X_tr[idx[0][0]], y_tr[idx[0][0]]
    print(X_tr.shape, y_tr.shape, len(X_dv), len(X_te))
    train = pd.DataFrame(data=np.hstack((X_tr, y_tr.reshape(len(y_tr), 1))))
    dev = pd.DataFrame(data=np.hstack((X_dv, y_dv.reshape(len(y_dv), 1))))
    test = pd.DataFrame(data=np.hstack((X_te, y_te.reshape(len(y_te), 1))))
    train.to_csv('data/train_split.csv')
    dev.to_csv('data/dev_split.csv')
    test.to_csv('data/test_split.csv')


if __name__ == '__main__':
    split_data()