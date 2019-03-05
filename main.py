import pandas as pd
import numpy as np

from model import build_model, fit_model, build_cnn_model, evaluate_model, predict
from utils import plot_learning, roc_callback, F1_score, auc_score

TRAIN_PATH = 'data/train_split.csv'
DEV_PATH = 'data/dev_split.csv'
TEST_PATH = 'data/test_split.csv'
HIDDEN_PATH = 'data/test.csv'
SPLIT = 0.8

def load_data(path):
    df = pd.read_csv(path)
    data = df.values
    X, y = data[:, 1:-1], data[:, -1]
    return (X, y)

def load_test_data(path):
    df = pd.read_csv(path)
    data = df.values
    X, y = data[:, 1:], data[:, 0]
    return (X, y)

def write_predictions(preds, labels, filename):
    labels = labels.reshape((len(labels), 1))
    df = pd.DataFrame(data=np.hstack((labels, preds)), columns=['ID_code', 'target'])
    df.set_index(keys='ID_code', inplace=True)
    df.to_csv(f'predictions/{filename}.csv')


def run():
    X_tr, y_tr = load_data(TRAIN_PATH)
    X_tr = X_tr.reshape((len(X_tr), X_tr.shape[1], 1))
    X_dv, y_dv = load_data(DEV_PATH)
    X_dv = X_dv.reshape((len(X_dv), X_dv.shape[1], 1))
    X_te, y_te = load_test_data(HIDDEN_PATH)
    X_te = X_te.reshape((len(X_te), X_te.shape[1], 1))
    cnn = build_cnn_model(X_tr.shape[1], 1)
    cnn.summary()
    cnn, hist = fit_model(cnn, 
                        X_tr[:int(len(X_tr)*SPLIT)], 
                        y_tr[:int(len(X_tr)*SPLIT)], 
                        X_tr[int(len(X_tr)*SPLIT):], 
                        y_tr[int(len(X_tr)*SPLIT):],
                        epochs=1, 
                        callbacks=[
                            roc_callback(
                                training_data=(X_tr[:int(len(X_tr)*SPLIT)], y_tr[:int(len(X_tr)*SPLIT)]), 
                                validation_data=(X_tr[int(len(X_tr)*SPLIT):], y_tr[int(len(X_tr)*SPLIT):])
                                )
                            ]
                        )
    preds_dv = predict(cnn, X_dv)
    auc = auc_score(y_dv, preds_dv)
    print(f'AUC score on development {auc}')
    plot_learning(hist.history)



if __name__ == '__main__':
    run()