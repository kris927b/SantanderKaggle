from keras.layers import Dense, Activation, Input, Conv1D, MaxPooling1D, Flatten
from keras.models import Model
import numpy as np


def build_model(in_dim, 
                no_classes, 
                loss='binary_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy']):
    
    in_layer = Input(shape=(in_dim, ), dtype='float32', name='Input')
    hidden_1 = Dense(128, name='Hidden1')(in_layer)
    relu_1 = Activation('relu', name='ReLU1')(hidden_1)
    hidden_2 = Dense(64, name='Hidden2')(relu_1)
    relu_2 = Activation('relu', name='ReLU2')(hidden_2)
    output = Dense(no_classes, name='Output')(relu_2)
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
    conv_1 = Conv1D(32, 5, name='Conv1')(in_layer)
    relu_1 = Activation('relu', name='ReLU1')(conv_1)
    conv_2 = Conv1D(16, 3, name='Conv2')(relu_1)
    relu_2 = Activation('relu', name='ReLU2')(conv_2)
    max_1 = MaxPooling1D(3, strides=2, name='MaxPool1')(relu_2)
    flat_1 = Flatten(name='Flatten1')(max_1)
    hidden_1 = Dense(32)(flat_1)
    relu_3 = Activation('relu')(hidden_1)
    output = Dense(no_classes, name='Output')(relu_3)
    output = Activation('sigmoid', name='Sigmoid1')(output)
    model = Model(in_layer, output)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def fit_model(model, X_tr, y_tr, X_val, y_val, callbacks=None):
    hist = model.fit(X_tr, y_tr, batch_size=64, epochs=2, validation_data=(X_val, y_val), callbacks=callbacks)
    return model, hist

def evaluate_model(model, X, y):
    loss, acc = model.evaluate(X, y, batch_size=len(X))
    print(f'Average loss {loss}; Average accuracy: {acc}')

def predict(model, X):
    return model.predict(X)