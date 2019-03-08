import pandas as pd
import numpy as np

from model import build_model, fit_model, build_cnn_model, evaluate_model, predict, fit_KFold
from utils import plot_learning, roc_callback, F1_score, auc_score

TRAIN_PATH = 'data/train_split.csv'
DEV_PATH = 'data/dev_split.csv'
TEST_PATH = 'data/test_split.csv'
HIDDEN_PATH = 'data/test.csv'
MODEL_PATH = 'models/final_model_fold_0.h5'
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
    # X_tr, y_tr = load_data(TRAIN_PATH)
    # X_tr = X_tr.reshape((len(X_tr), X_tr.shape[1], 1))
    # X_val, y_val = load_data(DEV_PATH)
    # X_val = X_val.reshape((len(X_val), X_val.shape[1], 1))
    # fit_KFold(X_tr.shape[1], 1, build_model, X_tr, y_tr, X_val, y_val)

    ### Predict and write predictions to file
    X_te, y_te = load_test_data(HIDDEN_PATH)
    X_te = X_te.reshape((len(X_te), X_te.shape[1], 1))
    model = build_model(X_te.shape[1], 1)
    model.load_weights(MODEL_PATH)
    preds = predict(model, X_te)
    write_predictions(preds, y_te, 'prediction')



if __name__ == '__main__':
    run()