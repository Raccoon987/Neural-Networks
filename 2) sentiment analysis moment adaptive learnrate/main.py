import numpy as np
import matplotlib.pyplot as plt

from preprocess import get_data, vectorizer, crossValidation, tokenize, stem_tokens
from sklearn.utils import shuffle
from itertools import product
import pandas as pd

from review_batch_moment_dropout import NeuralNetwork
#from review_batch_RMSProp import NeuralNetwork
#from review_batch_moment_RMSProp import NeuralNetwork
#from review_Adam import NeuralNetwork

def main():
    lst = [1768, 1555, 894, 674, 65, 54, 137, 327, 354, 492, 553, 634, 844, 928, 1054, 1118, 1228, 1449, 1474, 1483, 1504, 1529, 1559, 1733, 1881, 1917]
    data = get_data("products_sentiment_train.tsv", "products_sentiment_test_copy.tsv", balance=True, drop_lst=lst)
    #put here kwargs



    '''
    #layer_sequence = list(product(["tanh", "relu"], repeat=4))
    #layer_sequence = [(10, 10, 10, 10), (30, 30, 30, 30), (100, 100, 100, 100), (300, 300, 300, 300), (10, 30, 100, 300), (300, 100, 30, 10), (500, 150, 20, 2), (1000, 300, 30, 4),
    #                  (400, 100, 20, 10)]
    #layer_sequence = [10e-1, 10e-2, 10e-3, 10e-4, 10e-5]
    layer_sequence = list(product([1, 0], repeat=4))
    for index in range(len(layer_sequence) - 1):
        N = layer_sequence[index]
        #model = NeuralNetwork(400, 100, 20, 10, layer_sequence[index])
        #model = NeuralNetwork(N[0], N[1], N[2], N[3], ('relu', 'relu', 'tanh', 'relu'))
        model = NeuralNetwork(400, 100, 20, 10, ("tanh", "relu", "relu", "relu"), dropout=N, p=0.7)
        print(layer_sequence[index])
        scores = crossValidation(model, data, K=5, learning_rate=10e-5, reg=10e-2, epochs=251, show_fig=False)
        print("score mean:", np.mean(scores), "stdev:", np.std(scores))
    '''



    model = NeuralNetwork(400, 100, 20, 10, ("tanh", "relu", "relu", "relu"), dropout=(0, 1, 1, 0), p=0.7)
    train_size = (np.array(data[0]["text"])).shape[0]
    X, Y = vectorizer( np.append(np.array(data[0]["text"]), np.array(data[1]["text"])), tokenizer=tokenize, ngram_range=(1, 3), max_df=0.85, min_df=1, max_features=None), \
           np.array(data[0]["label"])
    model.fit(X[:train_size, :], Y, learning_rate=10e-5, reg=10e-2, epochs=231, show_fig=False)
    result = model.predict(X[train_size:, :])
    print(result)
    with open("neuro_1.csv", 'w') as f_out:
        f_out.write(pd.DataFrame(pd.Series(list(map(str, range(0, 500)))).str.cat(list(map(str, result)), sep=','),
                                 columns=["Id,y"]).to_csv(sep=" ", index=False))



    '''
    #("tanh", "relu", "relu", "relu")
    for i in range(3):
        model = NeuralNetwork(400, 100, 20, 10, ("tanh", "relu", "relu", "relu"), dropout=(0, 1, 1, 0), p=0.7)
        #X, Y = vectorizer(np.array(data[0]["text"]), ngram_range=(1, 3), max_df=0.85, min_df=1, max_features=None), np.array(data[0]["label"])
        #scores = crossValidation(model, data, K=5, batch_size=100, learning_rate=10e-6, reg=10e-6, epochs=501)

        #for batch_moment_learnrate learning_rate=5*10e-5, reg=10e-2 (-3) (-4), epochs=51
        #for batch_adaptive_learnrate: learning_rate=10e-4, reg=10e-2 (-3) (-4), epochs=151
        #for batch_moment: learning_rate=10e-5, reg=10e-2 (-3), epochs=251
        scores = crossValidation(model, data, K=5, learning_rate=10e-5, reg=10e-2, epochs=231, show_fig=False)
        print("relu, relu, tanh, relu ", "score mean:", np.mean(scores), "stdev:", np.std(scores))
    '''

if __name__ == '__main__':
    main()

