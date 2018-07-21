from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.utils
import keras.optimizers
from keras.preprocessing.text import text_to_word_sequence
from keras.backend import argmax

# define the document
import pandas as pd
from preprocess import get_data, error_rate, vectorizer, tokenize, stem_tokens
import operator
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np

from itertools import product

class KerasNeuralNetwork():
    def __init__(self, hidden_layer_sizes, nonlin_functions, dropout_coef):
        if (len(hidden_layer_sizes) != len(dropout_coef)) and (len(hidden_layer_sizes) != len(nonlin_functions)):
            print("LENGTH OF hidden_layer_sizes PARAMETERS MUST EQUAL TO LENGTH OF dropout_coef AND EQUAL TO LENGTH OF nonlin_functions")
            raise ValueError
        self.hidden_layers = hidden_layer_sizes
        self.nonlin_functions = nonlin_functions
        self.dropout_coef = dropout_coef

    #hidden_layers = property(operator.attrgetter('_hidden_layers'))
    #@hidden_layers.setter
    #def value(self, hidden_layer_sizes):
    #    if not all(layer_size in ["relu", "tanh", "softmax"] for layer_size in hidden_layer_sizes):
    #        raise Exception("'hidden_layer_sizes' parameter must be a list that contains 'relu', 'tanh' or 'softmax' elements")
    #    self._hidden_layers = hidden_layer_sizes

    def text_process_data(self, X, Y=np.array([])):
        X = X.astype(np.float32)
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        #X = X.astype(np.float32)
        if Y.any():
            Y = keras.utils.to_categorical(Y, num_classes=2).astype(np.float32)
            X, Y = shuffle(X, Y)
            Y_flat = np.argmax(Y, axis=1)
            #print(X.shape, Y.shape)
            return X, Y, Y_flat
        return X


    def init_model(self, D, K, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], custom_layers_sequence=None):
        print('Compiling Model ... ')

        # the model will be a sequence of layers
        model = Sequential()

        if custom_layers_sequence == None:
            for i in range(len(self.hidden_layers)):
                if i == 0:
                    model.add(Dense(units=self.hidden_layers[0], input_dim=D))
                else:
                    model.add(Dense(units=self.hidden_layers[i]))
                    model.add(Activation(self.nonlin_functions[i]))
                    model.add(Dropout(self.dropout_coef[i]))
            model.add(Dense(units=K))
            model.add(Activation('softmax'))
        else:
            '''later here may be implemented custom layer adding mechanism'''
            pass

        ''' list of losses: https://keras.io/losses/     list of optimizers: https://keras.io/optimizers/     list of metrics: https://keras.io/metrics/'''
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

    def fit_network(self, X, Y, model, epochs=20, batch=256, show=True):
        #X, Y, Y_flat = self.text_process_data(X, Y)
        try:
            print('Training model...')
            #print(type(X), type(Y))
            r = model.fit(X, Y, validation_split=0.25, epochs=epochs, batch_size=batch, verbose=2)

            if show:
                plt.figure(figsize=(12, 9))  # make separate figure
                plt.subplot(2, 1, 1)
                plt.plot(r.history['loss'], label='loss')
                plt.plot(r.history['val_loss'], label='val_loss')
                plt.legend()
                plt.grid()

                plt.subplot(2, 1, 2)
                plt.plot(r.history['acc'], label='acc')
                plt.plot(r.history['val_acc'], label='val_acc')
                plt.legend()
                plt.grid()
                plt.show()

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            return r, r.history['loss']

        return r

    def make_prediction(self, X, model, batch=32, verbose=0):
        X = self.text_process_data(X)
        return np.argmax(model.predict(X, batch_size=batch, verbose=verbose), axis=1)


def main_search(iter):
    lst = [1768, 1555, 894, 674, 65, 54, 137, 327, 354, 492, 553, 634, 844, 928, 1054, 1118, 1228, 1449, 1474, 1483,
           1504, 1529, 1559, 1733, 1881, 1917]
    data = get_data("products_sentiment_train.tsv", "products_sentiment_test_copy.tsv", balance=True, drop_lst=lst)
    X_train, Y_train = vectorizer(np.array(data[0]["text"]), tokenizer=tokenize, ngram_range=(1, 4), max_df=0.85, min_df=1, max_features=None), np.array(data[0]["label"])

    #network_class = KerasNeuralNetwork(hidden_layer_sizes=(400, 100, 20, 10), nonlin_functions=("tanh", "relu", "relu", "relu"), dropout_coef=(0.9, 0.8, 0.8, 0.8))
    network_class = KerasNeuralNetwork(hidden_layer_sizes=(400, 200, 10), nonlin_functions=("tanh", "tanh", "tanh"), dropout_coef=(0.7, 0.7, 0.7))
    X, Y, Y_flat = network_class.text_process_data(X_train, Y=Y_train)


    result = []
    for i in range(iter):
        neural_network = network_class.init_model(D=X.shape[1], K=Y.shape[1], loss='categorical_crossentropy',
                                                  optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0), metrics=['accuracy'])
        r = network_class.fit_network(X, Y, neural_network, epochs=30, batch=130, show=False)
        result.append(r)
    
    plt.figure(figsize=(12, 9))  # make separate figure
    plt.subplot(2, 1, 1)
    plt.plot(np.sum(np.array([r.history['loss'] for r in result]), axis=0)/iter, label='loss')
    plt.plot(np.sum(np.array([r.history['val_loss'] for r in result]), axis=0)/iter, label='val_loss')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(np.sum(np.array([r.history['acc'] for r in result]), axis=0)/iter, label='acc')
    plt.plot(np.sum(np.array([r.history['val_acc'] for r in result]), axis=0)/iter, label='val_acc')
    plt.legend()
    plt.grid()
    plt.show()

def main_predict():
    lst = [1768, 1555, 894, 674, 65, 54, 137, 327, 354, 492, 553, 634, 844, 928, 1054, 1118, 1228, 1449, 1474, 1483,
           1504, 1529, 1559, 1733, 1881, 1917]
    data = get_data("products_sentiment_train.tsv", "products_sentiment_test_copy.tsv", balance=True, drop_lst=lst)
    train_size = (np.array(data[0]["text"])).shape[0]
    X_full, Y_full = vectorizer(np.append(np.array(data[0]["text"]), np.array(data[1]["text"])), tokenizer=tokenize, ngram_range=(1, 4), max_df=0.85, min_df=1, max_features=None), np.array(data[0]["label"])
    network_class = KerasNeuralNetwork(hidden_layer_sizes=(400, 200, 10), nonlin_functions=("tanh", "tanh", "tanh"), dropout_coef=(0.7, 0.7, 0.7))
    X_train, Y_train, Y_flat_train = network_class.text_process_data(X_full[:train_size, :], Y=Y_full)
    neural_network = network_class.init_model(D=X_train.shape[1], K=Y_train.shape[1], loss='categorical_crossentropy',
                                              optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.99,  epsilon=1e-08, decay=0.0), metrics=['accuracy'])
    r = network_class.fit_network(X_train, Y_train, neural_network, epochs=30, batch=130, show=False)

    X_predict = network_class.text_process_data(X_full[train_size:, :])
    result = network_class.make_prediction(X_predict, neural_network, batch=130)
    with open("keras_adam.csv", 'w') as f_out:
        f_out.write(pd.DataFrame(pd.Series(map(str, range(0, 500))).str.cat(list(map(str, result)), sep=','),
                                 columns=["Id,y"]).to_csv(sep=" ", index=False))

#main_search(10)
main_predict()