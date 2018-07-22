In the third part of the Neural Network repository, I continue to build the neural network for the sentiment analysis problem. 
This time the code is written using the tensorflow and keras modules.


* preprocess.py - supplementary functions like cross-entropy function, nonlinear functions, function that gets and 
                  transform data and other

* products_sentiment_train.tsv,  products_sentiment_test_copy.tsv - train and test data with different reviews; taken 
                                                                    from "Yandex_sentiment_analysis" repository 

* keras_example.py - neural network written in keras; program allows you to plot dependencies for the loss function and 
                     accuracy value on the number of learning epochs for training and test data. In this case, averaging 
                     was performed (main_search function) - creation and fitting the model on the same parameters several 
                     times - to obtain a more accurate result. The graphics locate in the folder "keras review analysis 
                     result". According to the graphs, it is possible to determine the number of epochs, exceeding which 
                     occurs overfitting (the accuracy on training data increases and on the test data begins to fall down)  

* review_tf_save.py - neural network(NN) is implemented using tensorflow module, after fitting it saves session with all selected weights
                      into file on disk, then restore it to make a prediction

* review_tf_batchnorm_save.py - NN similat to previous but with batch normalization technique 

* review_tf_batchnorm.py - NN with batch normalization technique, it fit model and make a prediction without saving session to disk

* review_tensorflow_RESULTS.txt - accuracy results for different parameters and optimizators 