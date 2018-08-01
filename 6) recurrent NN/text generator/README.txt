RECURRENT NEURAL NETWORK (NN)


Programe similar to "make_poetry_univers.py" - generate text using recurrent NN. This time brown corpus was used as a 
learning text. Brown corpus - set of English-language texts on different subjects - political, sports, society and so on.

* TextGenerator.py - since the first layer of the neural network wis a word-embedding layer (each word from the vocabulary 
                     is mapped to vectors of real numbers), one of the goals of this program - retrieve weights of the
                     embedding layer after completion of the NN fitting process, reduce their dimension by means of TSNE or
                     PCA methods and make a two-dimensional plot of the result. If the NN model had a good convergence and
                     learned to reproduce sentences from train dataset - then dots on the graph that corresponds to similar 
                     in meaning words will be grouped into clusters. 

                     The next task was to find words of similar meaning for the sets of triples of words - for example 
                     ('france' - 'paris' = ? - 'london') or ('king' - 'man' = ? - 'woman'). Of course, to get a good result, 
                     you need to use a large and various training dataset of sentences. 
                     a - b = ? - c
                     Each word is represented by a vector. Unknown word will be the nearest vector-word from the whole training
                     dataset (using the Euclidean or cosine metrics).
                     index_sentence() and index_sentence_limit() functions return brown corpus sentences where each word is 
                     mapped with a number; second function leave only N most frequently used words, all other words are 
                     replaced by None.