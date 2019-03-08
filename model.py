from keras.layers import Dense, Activation, Input, Conv1D, MaxPooling1D, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import StratifiedKFold
import numpy as np

from utils import roc_callback, auc_score, plot_learning


def build_model(in_dim, 
                no_classes, 
                loss='binary_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy']):
    
    in_layer = Input(shape=(in_dim, 1), dtype='float32', name='Input')
    hidden_1 = Dense(64, name='Hidden1')(in_layer)
    relu_1 = Activation('relu', name='ReLU1')(hidden_1)
    flat = Flatten()(relu_1)
    output = Dense(no_classes, name='Output')(flat)
    output = Activation('sigmoid', name='Sigmoid1')(output)
    model = Model(in_layer, output)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_cnn_model(in_dim, 
                no_classes, 
                loss='binary_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy']):
    
    in_layer = Input(shape=(in_dim, 1), dtype='float32', name='Input')
    conv_1 = Conv1D(128, 5, name='Conv1')(in_layer)
    relu_1 = Activation('relu', name='ReLU1')(conv_1)
    conv_2 = Conv1D(64, 3, name='Conv2')(relu_1)
    relu_2 = Activation('relu', name='ReLU2')(conv_2)
    max_1 = MaxPooling1D(8, strides=2, name='MaxPool1')(relu_2)
    flat_1 = Flatten(name='Flatten1')(max_1)
    hidden_1 = Dense(32, name='Hidden1')(flat_1)
    relu_3 = Activation('relu', name='ReLU3')(hidden_1)
    hidden_2 = Dense(16, name='Hidden2')(relu_3)
    relu_4 = Activation('relu', name='ReLU4')(hidden_2)
    output = Dense(no_classes, name='Output')(relu_4)
    output = Activation('sigmoid', name='Sigmoid1')(output)
    model = Model(in_layer, output)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def fit_model(model, X_tr, y_tr, X_val, y_val, callbacks=None, epochs=10):
    hist = model.fit(X_tr, y_tr, batch_size=128, epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks)
    return model, hist

def evaluate_model(model, X, y):
    preds = predict(model, X)
    auc = auc_score(y, preds)
    return auc


def predict(model, X):
    return model.predict(X)

def get_callbacks(name, data_tr, data_val):
    save = ModelCheckpoint(name, save_best_only=True, monitor='val_roc_auc', mode='max')
    auc = roc_callback(data_tr, data_val)
    return [auc, save]

def fit_KFold(in_dim, no_classes, model_fn, X, y, X_val, y_val, K=5):
    folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=1).split(X, y))

    for i, (idx_tr, idx_val) in enumerate(folds):
        print(f'\nFold: {i}')
        data_tr = (X[idx_tr], y[idx_tr])
        data_val = (X[idx_val], y[idx_val])

        name = f'models/final_model_fold_{i}.h5'
        callbacks = get_callbacks(name, data_tr, data_val)
        model = model_fn(in_dim, no_classes)
        model, hist = fit_model(model, data_tr[0], data_tr[1], data_val[0], data_val[1], callbacks=callbacks, epochs=30)
        plot_learning(hist.history, i)

        auc = evaluate_model(model, X_val, y_val)
        print(f'AUC score for fold {i}: {auc}')


