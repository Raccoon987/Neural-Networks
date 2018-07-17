import numpy as np
import matplotlib.pyplot as plt

from preprocess import get_data, vectorizer, crossValidation, tokenize, stem_tokens
from sklearn.utils import shuffle
from itertools import product
import pandas as pd
from review_neural_2 import NeuralNetwork


f = lambda f_name: os.path.realpath(os.path.join(os.getcwd(), f_name)).replace('\\', '/')

def main():
    lst = [1768, 1555, 894, 674, 65, 54, 137, 327, 354, 492, 553, 634, 844, 928, 1054, 1118, 1228, 1449, 1474, 1483, 
           1504, 1529, 1559, 1733, 1881, 1917]
    data = get_data(f("products_sentiment_train.tsv"), f("products_sentiment_test_copy.tsv"), balance=True, drop_lst=lst)
    #put here kwargs

    ''' try to find best combination of layer nonlinear functions '''
    #layer_sequence = list(product(["tanh", "relu"], repeat=4))
    #for index in range(len(layer_sequence)):
    #    model = NeuralNetwork(400, 100, 20, 10, layer_sequence[index])

    #    scores = crossValidation(model, data, K=5, learning_rate=10e-6, reg=10e-7, epochs=10001)
    #    print(layer_sequence[index])
    #    print("score mean:", np.mean(scores), "stdev:", np.std(scores))

    model = NeuralNetwork(400, 100, 20, 10, ("tanh", "relu", "relu", "relu"))
    train_size = (np.array(data[0]["text"])).shape[0]
    X, Y = vectorizer(np.append(np.array(data[0]["text"]), np.array(data[1]["text"])), tokenizer=tokenize, ngram_range=(1, 3), \
                      max_df=0.85, min_df=1, max_features=None), np.array(data[0]["label"])
    model.fit(X[:train_size, :], Y, learning_rate=10e-6, reg=10e-7, epochs=11001)
    result = model.predict(X[train_size:, :])
    print(result)
    with open(f("neuro_1.csv"), 'w') as f_out:
        f_out.write(pd.DataFrame(pd.Series(map(str, range(0, 500))).str.cat(map(str, result), sep=','),
                                 columns=["Id,y"]).to_csv(sep=" ", index=False))
    #for i in range(3):
    #    model = NeuralNetwork(400, 100, 20, 10, ("tanh", "relu", "relu", "relu"))
    #    #X, Y = vectorizer(np.array(data[0]["text"]), ngram_range=(1, 3), max_df=0.85, min_df=1, max_features=None), np.array(data[0]["label"])
    #    scores = crossValidation(model, data, K=5, learning_rate=10e-6, reg=10e-7, epochs=11001)
    #    print("tanh, relu, relu, relu ", "score mean:", np.mean(scores), "stdev:", np.std(scores))

if __name__ == '__main__':
    main()

