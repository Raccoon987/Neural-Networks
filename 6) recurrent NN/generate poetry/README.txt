IMPLEMENTATION OF RECURRENT NEURAL NETWORK (NN)


1) Poetry generator

* make_poetry_univers.py - this programe learn how to generate poetry from given text file with poems. Here we use word embedding technique, 
                           when each word transform into a D dimensional vector. So initial one-hot encoded word with dimension V (V - size 
                           of vocabulary - all words from train data) turn into D dimensional vector. For this reason an additional first 
                           embedding layer in NN is created. To get weights for this first new layer the same gradient descent method is used.
                           
                           get_poetry_data() function returns file text converted into unique indices (each different word has its own unique
                           index) and mapping {word: index}
                           four different recurrent units may be utilized - ordinary, rate or gate recurrent units and long-short-term-memorry 
                           (LSTM).
                           For each line of prediction model try to find the best new word for already given previous words. During "learning"
                           phase, NN network try to reproduce lines from train data file with poetry. Each line starts with word "START" and 
                           some of them ends with word "END". For each sentence, one of the two cases is randomly chosen - to predict the 
                           sentence to its last word or to the word "END", which is added at the end of the sentence. This is done to shorten 
                           the length of predictions. Cost function may be plotted and saved into a image file. Pretrained NN weights are saved
                           into .npz file - training NN may take a lot of time.  
                           generate() function - NN model creates 4 lines of text


* edgar_allan_poe.txt - file with poetry text used for training NN

* LSTM.npz - pretrained NN weights based on LSTM recurrent unit

* .png - plotted and saved log(cost) function