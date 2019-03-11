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


def data_augmentation():
    df_train = pd.read_csv('data/train.csv')
    X_gen = df_train.values[:, 2:]
    y_gen = df_train.values[:, 1]

    df_train_ones = df_train.loc[df_train['target'] == 1]
    
    X = df_train_ones.values[:, 2:]
    y = df_train_ones.values[:, 1]

    mean_point = X.mean(axis=0)
    std_point = np.zeros(shape=mean_point.shape)
    for feature in range(len(X[0])):
        f = list(X[:,feature])
        std_point[feature] = np.std(f)

    number_of_new_points = 100000
    new_x = np.zeros(shape=(number_of_new_points, mean_point.shape[0]))
    new_y = np.ones(shape=(number_of_new_points,))
    for i in range(number_of_new_points):
        for j in range(mean_point.shape[0]):
            new_x[i][j] = np.random.normal(mean_point[j], std_point[j])

    X = np.concatenate((X, new_x), axis=0)
    y = np.concatenate((y, new_y), axis=0)    

    X_gen = np.concatenate((X_gen, X), axis=0)
    y_gen = np.concatenate((y_gen, y), axis=0)


    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
    idx = list(sss.split(X_gen, y_gen))
    X_tr, y_tr = X_gen[idx[0][0]], y_gen[idx[0][0]]
    X_dv, y_dv = X_gen[idx[0][1]], y_gen[idx[0][1]]
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
    #prop()
    #data_augmentation()