import numpy as np
import pandas as pd
import csv, scipy
from sklearn.utils import shuffle
import nltk.corpus
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer, RSLPStemmer, WordNetLemmatizer

#LancasterStemmer(), RSLPStemmer(), SnowballStemmer("english", ignore_stopwords=False), PorterStemmer(), WordNetLemmatizer()
stemmer = LancasterStemmer()

#initialize start weights and biases
def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2)/np.sqrt(M1 + M2)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)

#all nonlinear functions
def relu(x):
    return x * (x > 0)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def softmax(x):
    return  (np.exp(x) * 1.0) / (np.exp(x)).sum(axis=1, keepdims=True)

#def tanh(x):
#    return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)

#cross-entropy function T - class Y - probability
def cost(T, Y):
    #print("T: ", T.shape, "Y: ", Y.shape)
    #print(type(T), type(Y))
    #print(T[0], Y[0])
    return -(T * np.log(Y)).sum()

def cost2(T, Y):
    # same as cost(), just uses the targets to index Y
    # instead of multiplying by a large indicator matrix with mostly 0s
    N = len(T)
    return -np.log(Y[np.arange(N), T]).mean()

def error_rate(targets, predictions):
    return np.mean(targets != predictions)

def y_hot_encoding(y):
    N = len(y)
    K = len(set(y))
    matrix = np.zeros((N, K))
    for i in range(N):
        matrix[i, y[i]] = 1
    return matrix

#FOR BINARY CLASSIFICATION
def get_data(train_file, test_file, balance=True, drop_lst=False):
    #text reviews
    train = pd.read_csv(train_file, names=["text", "label"], header=0, sep="\t")
    test = pd.read_csv(test_file, header=0, sep="\t", quoting=csv.QUOTE_NONE)
    del test["Id"]

    if drop_lst:
        train = train.drop(train.index[drop_lst])

    if balance:
        ones, zeros = (train[train["label"] == 1].shape)[0], (train[train["label"] == 0].shape)[0]
        if ones > zeros:
            balance_label = 0
        else:
            balance_label = 1

        balance_class = train[train["label"] == balance_label]
        train = shuffle(train.append(train.iloc[list(balance_class.index[:np.absolute(ones - zeros)])]), random_state=10)

    return train, test


#http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text, stemmer=PorterStemmer()):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer=stemmer)
    return stems

# kwargs: encoding, decode_error, strip_accents, lowercase, preprocessor, tokenizer, analyzer, stop_words, token_pattern,
# ngram_range, max_df, min_df, max_features, vocabulary, binary, dtype, norm, use_idf, smooth_idf, sublinear_tf
def vectorizer(data, **kwargs):
    tfidf_vect = TfidfVectorizer(kwargs)
    return tfidf_vect.fit_transform(data)

def crossValidation(model, data, batch_size, K=5, learning_rate=10e-6, reg=10e-7, epochs=10001, show_fig=False):
    # split data into K parts
    X, Y = vectorizer(np.array(data[0]["text"]),  
		      tokenizer=tokenize, 
		      ngram_range=(1, 3), 
		      max_df=0.85, 
		      min_df=1, 
		      max_features=None), \
	   np.array(data[0]["label"])

    #X, Y = vectorizer(np.array(data[0]["text"]), 
    #		       ngram_range=(1, 3), 
    #		       max_df=0.85, 
    #		       min_df=1, 
    #		       max_features=None), \
    #	    np.array(data[0]["label"])
     
    sz = (X.shape)[0] // K
    errors = []
    for k in range(K):
        model.fit(scipy.sparse.vstack((X[:k * sz, :], X[(k * sz + sz):, :])), np.append(Y[:k * sz], Y[(k * sz + sz):]), 
		  batch_size=batch_size,
                  learning_rate=learning_rate, 
		  reg=reg, 
		  epochs=epochs, 
		  show_fig=show_fig)
        
	err = model.score(X[k * sz:(k * sz + sz), :], Y[k * sz:(k * sz + sz)])
        print(err)
        errors.append(err)
    print("errors:", errors)
    return np.mean(errors)

