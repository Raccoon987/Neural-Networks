Regular feed-forward neural network with four fully connected hidden layers written from scratch with standard 
backpropagation algorithm implemented without using any neural libraries BUT using some technics like:

- RMSProp
- momentum
- momentum and dropout 
 + using batches

The task of this network is the same - sentiment analysis - binary classification of product reviews from 
"Yandex_sentiment_analysis" repository. 

Some math that explaines implemented technics that improve neural network performance and convergence rate - look in file
... 


* review_neural_batch_RMSProp.py - neural network implementation using batches and special decreasing learning rate technic

* 

* 



* preprocess.py - supplementary functions like cross-entropy function, nonlinear functions, function that gets and 
transform data and other

* explore_data.py - functions that were used to explore data, its distribution over review length and seach for reviews 
that prevent classifier from showing a good result 

* main.py - run all files in sertain order, train network and get classification result

* products_sentiment_train.tsv,  products_sentiment_test_copy.tsv - train and test data with different reviews; taken 
from "Yandex_sentiment_analysis" repository  

* neuro_1.csv - answers given by classifier