AUTOENCODERS

This folder contain autoencoder implementation. Pretraining neural networks - compress data to the hidden layer representation
and then try to reconstruct it into smth. that closely matches original data. It is an attempt to learn most important features 
and to discard noise. 

Autoencoders, Restricted Bolzman machines, Convolutional autoencoders; deep neural networks and deep belief networks will be 
considered.

Deep neural network consist of stacked autoencoders, when hidden layer of previous autoencoder become an input of next autoencoder.
So the data is reconstructuted only by the last encoder element. If Restricted Bolzman machines is stacked together than we create 
deep belief network. 

* autoencoders.py - implementation of Autoencoder, Restricted Bolzman machine, Convolutional autoencoder; 
                    deep neural networks (deep belief networks)

* main_autoencoders.py - functions that import classes from autoencoders.py and try to reconstruct images with human faces or
                         mnist numerals

* conv_autoenc_vanila.py - neural network for face emotion recognition with structure: 
                           n convolution layers => deep neural network (several autoencoders) => regular fully-connected layer

* mnist_train.csv - mnist numerals dataset

* several face images and their reconstructuction