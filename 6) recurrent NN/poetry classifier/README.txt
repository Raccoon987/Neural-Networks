IMPLEMENTATION OF RECURRENT NEURAL NETWORK


1) Poetry classification

* ClassPoetry.py - poetry classification. The program was fitted to distinguish text written by different poets - Edgar Allan Poe,
                   Robert Frost and William Shakespeare. The problem of binary (Edgar Allan Poe,Robert Frost) and three-class 
                   classification was considered. 
                   Each line of text has been converted into sequense of parts of speech. For this purpose functions convert_word_to_tag 
                   and get_poetry_classifier_data were used. Each part of the speech was assigned an index and the original text was 
                   converted into a sequence of indices. 
                   Four different recurrent units were implemented Simple recurrent unit and three more sophisticated ones - rate recurrent 
                   unit, gate recurrent unit and long short term memory. 
                   After learning phase, weights of the neural network are saved to a file. 

* poe.txt, robert_frost.txt, shakespeare.txt - files with poetry

* poe_frost.npz, poe_shakespeare_frost.npz - files with array that contain poetry converted into a parts of speech indices, author labels 
                                             and indices-parts_of_speech mapping

* poe_shakes_frost_verify.txt, poe_frost_verify.txt, poe_frost_verify.npz - similar files for testing the quality of the model

* LSTMPoeFrost_adam10-4_0.9_150ep.npz - saved pretrained weights for binary poe/frost poetry classification; adam optimizer, learn_rate=10-4,
                                        beta1=0.9, train for 150 epochs. 


 