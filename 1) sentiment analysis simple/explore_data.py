from preprocess import get_data, tokenize, stem_tokens, vectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import  CountVectorizer, TfidfVectorizer

train, test = get_data("products_sentiment_train.tsv", "products_sentiment_test_copy.tsv", balance=False)

def fun(data):
    cvect = CountVectorizer()
    token_matrix = cvect.fit_transform(data["text"])
    pd_token_matrix = pd.DataFrame(token_matrix.A, columns=cvect.get_feature_names())

    # Summarize review length
    print("review length: ")
    result = [pd_token_matrix.iloc[index].sum() for index, row in pd_token_matrix.iterrows()]
    print("rewiev mean length %.2f with dispersion (%f)" % (np.mean(result), np.std(result)))
    result.sort(reverse=False)
    print(result)

    idx = [index for index, row in pd_token_matrix.iterrows() if pd_token_matrix.iloc[index].sum() >= 50]
    for i in idx:
        #print(train["text"].iloc[i], i)
        print(i)
    # plot review length
    plt.boxplot(result)
    plt.grid()
    plt.show()


def check_tokenizer():
    #check does tokenizer "cut" words
    tfidf = TfidfVectorizer(tokenizer=tokenize)
    frequency_counts = tfidf.fit_transform(train["text"])
    matrix = pd.DataFrame(frequency_counts.A, columns=tfidf.get_feature_names())
    print(matrix)

# 1768, 1555, 894, 674, 65 - bad reviews
# 54, 137, 327, 354, 492, 553, 634, 844, 928, 1054, 1118, 1228, 1449, 1474, 1483, 1504, 1529, 1559, 1733, 1881, 1917 - too long reviews
#fun(test)
