Sequence-to-sequence architecture - TRANSLATOR

This neural network was created for sentence translation.

LSTM_1(return_sequence=False) - from the encoder part of network only takes h and c vector of final hidden states - small vector
representation of original sentence. 

LSTM_2 from the decoder part of network has the same size as LSTM_1 and use h and c as initial state

Here we use teacher forcing - each LSTM_2 unit gets as input not previously predicted word but actual next word of the translated 
sentence - for training this works better. 

-----------------------------------------------------------------------------------------------------

TRAINING 
	 	 			 -----------------------------------           
					|                                   |         
					|                                   |         
					|    ^            ^            ^    |         
					|    |            |            |    |         
					| -------      -------      ------- |         
Input_original => Embedding_original =>	|| LSTM_1| => | LSTM_1| => | LSTM_1|| => h => |
 (length Tx)				| -------      -------      ------- |    c    | 
					|    ^            ^            ^    |         |
					|    |            |            |    |         | 
					|   The          dog         barks  |         |
					|                                   |         |
		 	 		 -----------------------------------          |
         		 			    Encoder RNN                       |    
                                                              			      | 
										      |	
					    ------initial state for LSTM 2------------
					   |
					   |
					   |
                         		   |	 -------------------------------------------------------------
					   |	| <first word   <second word    <third word                   |
					   |	|   predict>      predict>        predict>        <end_token> |
					   |	|      ^             ^                ^                ^      |
					   |	|      |    h_1      |     h_2        |     ...        |      |
					   |	|  -------  c_1  -------   c_2    -------   ...    -------    |
Input_translate => Embedding_translate  => =>	| | LSTM_2|  => | LSTM_2|   =>   | LSTM_2|   =>   | LSTM_2|   | => Dense layer
 (length Ty)					|  -------       -------          -------          -------    |
						|     ^             ^                ^                ^       |
						|     |             |                |                |       |
						|<start_token>  <real first      <real second     <real third |
						|                   word             word             word    |
						|               translation>      translation>    translation>|
			 			 -------------------------------------------------------------
					       				Decoder RNN 




PREDICTION 


	<first predicted word>   	     <second predicted word>
		 ^				       ^
		 |				       |	
              -------          			    -------		
  h,C =>     | LSTM_2|            => h_1, c_1 =>   | LSTM_2|            => h_1, c_1 => ... => <end_token>
	      ------- 				    -------
		 ^				       ^
		 |				       |	
	   Input_translate = <start_token>	<first predicted
            (length = 1)			      word>

-----------------------------------------------------------------------------------------------------------------

translator_preprocess.py - takes configuration file with structure of neural network, txt file with sentences in two languages, 
                           and files with pretrained embedding word vectors. Returns lists with tokenized origin and translation 
                           sentences, embedding matricies, word_to_index dictionaries and modified configuration file

translator_model.py - KerasModel class that helps to build keras.Model from configuration file - initialize, compile and train it.
                      fit class method use fit_generator keras method that enables to train model on large datasets;
		      model_initializer method restore and connect neural network layers from configuration file.

translator_main.py - create three keras models for training and making translation;
                     wiki.ru.vec file with russian pretrained embedding vectors was taken from https://github.com/chakki-works/chakin
 		     glove.6B.100d.txt - english pretrained embedding vectors from https://nlp.stanford.edu/projects/glove/
		     en_rus_translator.txt with english-russian sentences from http://www.manythings.org/anki/ 	 

translator_train.py, translator_predict.py - configuration files with neural network structure. specify structure and all information
        				     about neural network here and feed (pass) it to KerasModel class.  