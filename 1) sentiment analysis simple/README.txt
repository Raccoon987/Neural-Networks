Regular feed-forward neural network with four fully connected hidden layers written from scratch with standard 
backpropagation algorithm implemented without using any neural libraries and without any "fancy stuff".  

Task of this network - sentiment analysis - binary classification of product reviews from "Yandex_sentiment_analysis"
repository. 


* review_neural_2.py - neural network implementation

* preprocess.py - supplementary functions like cross-entropy function, nonlinear functions, function that gets and 
transform data and other

* explore_data.py - functions that were used to explore data, its distribution over review length and seach for reviews 
that prevent classifier from showing a good result 

* main.py - run all files in sertain order, train network and get classification result

* products_sentiment_train.tsv,  products_sentiment_test_copy.tsv - train and test data with different reviews; taken 
from "Yandex_sentiment_analysis" repository  

* neuro_1.csv - answers given by classifier



