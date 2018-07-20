Regular feed-forward neural network with four fully connected hidden layers written from scratch with standard 
backpropagation algorithm implemented without using any neural libraries BUT using some technics like:

- RMSProp
- momentum
- momentum and dropout 
 + using batches

The task of this network is the same - sentiment analysis - binary classification of product reviews from 
"Yandex_sentiment_analysis" repository. 

Some math that explaines implemented technics for improvement neural network performance and convergence rate - look in file
some_math.jpg 


* review_batch_RMSProp.py - neural network implementation using batches and special decreasing learning rate technic

* review_batch_momentum_RMSProp.py - neural network implementation using batches and special decreasing learning rate 
technic + simple momentum

* review_Adam.py - Adam optimizer (adaptive moment estimation) implementation

* review_batch_momentum_dropout.py - same neural network with introducing simple momentum and dropout technic



* preprocess.py - supplementary functions like cross-entropy function, nonlinear functions, function that gets and 
transform data and other

* explore_data.py - functions that were used to explore data, its distribution over review length and seach for reviews 
that prevent classifier from showing a good result 

* main.py - run all files in sertain order, train network and get classification result

* products_sentiment_train.tsv,  products_sentiment_test_copy.tsv - train and test data with different reviews; taken 
from "Yandex_sentiment_analysis" repository  

* neuro_1.csv - answers given by classifier