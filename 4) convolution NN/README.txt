Convolution neural network for pictures classification. "fer2013.csv" file contains images with human faces expressing certain emotions - 
anger, fear, joy, disgust ... this file was downloaded from kaggle competition - 
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

* preprocess.py - contains supporting functions that get and convert data into a suitable form, initialize weights and performe one_hot_encoding 
                  operation

* conv_tf_face_recognition.py - main file with convolution neural network (NN) implementation using tensorflow. First two classes create ordinary 
                                fully-connected layers; second one class has batch normaliztion technic. Next two classes - convolution layer and 
                                unrealized for this moment convolution layer with batch normalization. Last class - class that creates NN, according 
                                to a given number of layers and parameters, trains it and makes a prediction. 

* explore_face_recognition.py - attempt to find best parameters. several similar functions try to find best combination of nonlinearn functions, 
                                number of convolution and ordinary layers, optimization technics, learning rate and other. All results are saved 
                                into csv files and then may be plotted by means of plot_() function to compare results.

* conv_face_grid_search  - folder contain results and graphs

* fer2013.csv - file with images data 