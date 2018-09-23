import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.utils import shuffle
import difflib
from nltk import pos_tag, word_tokenize
from nltk.tokenize import RegexpTokenizer


f = lambda f_name: os.path.realpath(os.path.join(os.getcwd(), f_name)).replace('\\', '/')

def init_weight_and_bias(M1, M2):
    #return (np.random.randn(M1, M2) * np.sqrt(2.0 / M1)).astype(np.float32), (np.zeros(M2)).astype(np.float32)
    return (np.random.randn(M1, M2) / np.sqrt(M2 + M1)).astype(np.float32), (np.zeros(M2)).astype(np.float32)

#def remove_punctuation(sentence):
#    return sentence.translate(str.maketrans('', '', string.punctuation))

def convert_word_to_tag(s):
    ''' word_tokenize(string) => [list of words]; nltk.pos_tag([list of words]) => [list of tuples (word, part_of_speech_noun_verb...)]  '''
    tokenizer = RegexpTokenizer(r'\w+')
    word_partOfSpeech = pos_tag(tokenizer.tokenize(s))
    return [y for x, y in word_partOfSpeech]

def x_hot_encoding(x, enc_len):
    N = len(x)
    matrix = np.zeros((N, enc_len))
    for i in range(N):
        matrix[i, x[i]] = 1
    return matrix

def get_poetry_classifier_data(file_lst, filename):
    word2idx = {}
    X_sentences, Y_label = [], []
    for file, label in zip(file_lst, range(len(file_lst))):
        for line in open(f(file)):
            if line:
                tags = convert_word_to_tag(line)
                if len(tags) > 1:
                    for tag in tags:
                        if tag not in word2idx:
                            word2idx[tag] = len(word2idx)
                    X_sentences.append([word2idx[i] for i in tags])
                    Y_label.append(label)

            ''' SO WE RETURN LISTS WITH INDEXIS CORRESPONDS TO PARTS OF SPEECH AND LABELS CORRESPONDS TO AUTHOR OF CURRENT STRING '''
    np.savez(filename, X_sentences, Y_label, len(word2idx))
    return np.asarray(X_sentences), np.asarray(Y_label), word2idx



class SimpleReccUnit():
    
    def __init__(self, D, M2, nonlin_func, Wb_npz=None):
        self.D, self.M2 = D, M2
        self.nonlin_func = nonlin_func
        self.set_Wb(Wb_npz)


    def set_Wb(self, Wb_npz):
        if Wb_npz:
            self.Wx_h, self.Wh_h = [list(map(tf.Variable, W_b)) for W_b in Wb_npz]
        else:
            self.Wx_h, self.Wh_h = [self.helper(dim) for dim in ((self.D, self.M2), 
                                                                 (self.M2, self.M2))]

        self.params = [self.Wx_h, self.Wh_h]


    def get_params(self):
        return self.params


    def helper(self, dim):
        return list(map(tf.Variable, init_weight_and_bias(dim[0], dim[1])))


    def nonlinear(self, func):
        nonlin_dict = {"relu" : tf.nn.relu, 
                       "tanh" : tf.nn.tanh, 
                       "softmax" : tf.nn.softmax, 
                       "sigmoid" : tf.nn.sigmoid, 
                       "None" : lambda x: x}

        if func in nonlin_dict.keys():
            return nonlin_dict[func]


    def recurrence(self, prev_h_rec, current_X):
        current_X = tf.reshape(current_X, (1, self.D))
        prev_h_rec = tf.reshape(prev_h_rec, (1, self.M2))

        h_recur = self.nonlinear(self.nonlin_func)(  
                                                   tf.matmul(current_X, self.Wx_h[0]) + \
                                                   tf.matmul(tf.reshape(prev_h_rec, (1, self.M2)), self.Wh_h[0]) + \
                                                   self.Wh_h[1])
        return tf.reshape(h_recur, (self.M2,))

    def output(self, Xw):
        h_hidden = tf.scan(fn=self.recurrence,                  # function to apply to each elems
                           elems=Xw,                            # fn takes each of element in elems
                           initializer=self.Wx_h[1], )
        return h_hidden



class RateReccUnit():
    
    def __init__(self, D, M2, nonlin_func, Wb_npz=None):
        self.D, self.M2 = D, M2
        self.nonlin_func = nonlin_func
        self.set_Wb(Wb_npz)


    def set_Wb(self, Wb_npz):
        if Wb_npz:
            self.Wx_h, self.Wx_z, self.Wh_z, self.Wh_h = [list(map(tf.Variable, W_b)) for W_b in Wb_npz]
        else:
            self.Wx_h, self.Wx_z, self.Wh_z, self.Wh_h = [self.helper(dim) for dim in ((self.D, self.M2), 
										       (self.D, self.M2), 
                                                                                       (self.M2, self.M2), 
										       (self.M2, self.M2))]

        self.params = [self.Wx_h, self.Wx_z, self.Wh_z, self.Wh_h]


    def get_params(self):
        return self.params


    '''          =====================================       '''
    ''' {X} => || hHat_rec ->     z_g   ->  h_rec     ||    '''
    '''        ||      ^                    ^   |     ||     '''
    '''        ||      |        1 - z_g -> /    |     ||     '''
    '''        ||      |            ^           |     ||     '''
    '''        ||      |____________|___________|     ||     '''
    '''z_g = sigm(x.dot(Wx_z) + h_rec(t-1).dot(Whrec_zg) +bz)'''
    '''hHat_rec = f(x.dot(Wx_h_rec) + r * h_rec(t-1).dot(W_rec) + brec)'''
    '''h_rec = (1 - z_g) * h_rec(t-1) + z_g * hHat_rec'''

    def helper(self, dim):
        return list(map(tf.Variable, init_weight_and_bias(dim[0], dim[1])))

    def nonlinear(self, func):
        nonlin_dict = {"relu" : tf.nn.relu, 
		       "tanh" : tf.nn.tanh, 
		       "softmax" : tf.nn.softmax, 
		       "sigmoid" : tf.nn.sigmoid, 
		       "None" : lambda x: x}

        if func in nonlin_dict.keys():
            return nonlin_dict[func]


    def recurrence(self, prev_h_rec, current_X):
        current_X = tf.reshape(current_X, (1, self.D))
        prev_h_rec = tf.reshape(prev_h_rec, (1, self.M2))

        hHat_recur = self.nonlinear(self.nonlin_func)(
 						      tf.matmul(current_X, self.Wx_h[0]) + \
                                                      tf.matmul(prev_h_rec, self.Wh_h[0]) + \
                                                      self.Wh_h[1])
        
	z_gate = self.nonlinear("sigmoid")(
					   tf.matmul(current_X, self.Wx_z[0]) + \
					   tf.matmul(prev_h_rec, self.Wh_z[0]) + \
					   self.Wh_z[1])

        h_recur = (1 - z_gate) * prev_h_rec + z_gate * hHat_recur

        return tf.reshape(h_recur, (self.M2,))


    def output(self, Xw):
        h_hidden = tf.scan(fn=self.recurrence,  		# function to apply to each elems
                           elems=Xw,  				# fn takes each of element in elems
                           initializer=self.Wx_h[1], )
        return h_hidden


class GateReccUnit():
    def __init__(self, D, M2, nonlin_func, Wb_npz=None):
        self.D, self.M2 = D, M2
        self.nonlin_func = nonlin_func
        self.set_Wb(Wb_npz)

    def set_Wb(self, Wb_npz):
        if Wb_npz:
            self.Wx_h, self.Wx_z, self.Wh_z, self.Wh_h, self.Wx_r, self.Wh_r = [list(map(tf.Variable, W_b)) for W_b in Wb_npz]
        else:
            self.Wx_h, self.Wx_z, self.Wh_z, self.Wh_h, self.Wx_r, self.Wh_r = [self.helper(dim) for dim in ((self.D, self.M2), 
													     (self.D, self.M2), 
													     (self.M2, self.M2), 
                                                                                 			     (self.M2, self.M2), 
													     (self.D, self.M2), 
													     (self.M2, self.M2))]

        self.params = [self.Wx_h, self.Wx_z, self.Wh_z, self.Wh_h, self.Wx_r, self.Wh_r]



    def get_params(self):
        return self.params

    '''          =====================================    '''
    ''' {X} => || hHat_rec ->   z_g    ->  h_rec      ||  '''
    '''        ||      ^                    ^   |     ||  '''
    '''        ||      r        1 - z_g -> /    |     ||  '''
    '''        ||      ^             ^          |     ||  '''
    '''        ||      |_____________|__________|     ||  '''
    '''r = sigm(x.dot(Wx_r) + h_rec(t-1).dot(Whrec_r) +br) '''
    '''z_g = sigm(x.dot(Wx_z) + h_rec(t-1).dot(Whrec_zg) +bz)'''
    '''hHat_rec = f(x.dot(Wx_h_rec) + r * h_rec(t-1).dot(W_rec) + brec)'''
    '''h_rec = (1 - z_g) * h_rec(t-1) + z_g * hHat_rec'''

    def helper(self, dim):
        return list(map(tf.Variable, init_weight_and_bias(dim[0], dim[1])))

    def nonlinear(self, func):
        nonlin_dict = {"relu" : tf.nn.relu, 
		       "tanh" : tf.nn.tanh, 
		       "softmax" : tf.nn.softmax, 
		       "sigmoid" : tf.nn.sigmoid, 
		       "None" : lambda x: x}

        if func in nonlin_dict.keys():
            return nonlin_dict[func]

    def recurrence(self, prev_h_rec, current_X):
        current_X = tf.reshape(current_X, (1, self.D))
        prev_h_rec = tf.reshape(prev_h_rec, (1, self.M2))

        r = self.nonlinear("sigmoid")(
                                      tf.matmul(current_X, self.Wx_r[0]) + \
				      tf.matmul(prev_h_rec, self.Wh_r[0]) + \
				      self.Wh_r[1])

        z_gate = self.nonlinear("sigmoid")(
					   tf.matmul(current_X, self.Wx_z[0]) + \
					   tf.matmul(prev_h_rec, self.Wh_z[0]) + \
					   self.Wh_z[1])

        hHat_recur = self.nonlinear(self.nonlin_func)(
						      tf.matmul(current_X, self.Wx_h[0]) + \
						      r * tf.matmul(prev_h_rec, self.Wh_h[0]) + \
						      self.Wh_h[1])

        h_recur = (1 - z_gate) * prev_h_rec + z_gate * hHat_recur

        return tf.reshape(h_recur, (self.M2,))

    def output(self, Xw):
        h_hidden = tf.scan(fn=self.recurrence,  		# function to apply to each elems
                           elems=Xw,  				# fn takes each of element in elems
                           initializer=self.Wx_h[1], )
        return h_hidden

class LSTM():
    def __init__(self, D, M2, nonlin_func, Wb_npz=None):
        self.D, self.M2 = D, M2
        self.nonlin_func = nonlin_func
        self.set_Wb(Wb_npz)

    def set_Wb(self, Wb_npz):
        if Wb_npz:
            self.Wx_i, self.Wc_i, self.Wh_i, self.Wx_f, self.Wc_f, self.Wh_f, self.Wx_c, self.Wh_c, self.Wx_o, self.Wc_o, self.Wh_o = \
                                  [list(map(tf.Variable, W_b)) for W_b in Wb_npz]
        else:
            self.Wx_i, self.Wc_i, self.Wh_i = [self.helper(dim) for dim in ((self.D, self.M2), 
									    (self.M2, self.M2), 
									    (self.M2, self.M2))]

            self.Wx_f, self.Wc_f, self.Wh_f = [self.helper(dim) for dim in ((self.D, self.M2), 
									    (self.M2, self.M2), 
									    (self.M2, self.M2))]

            self.Wx_c, self.Wh_c = [self.helper(dim) for dim in ((self.D, self.M2), 
								 (self.M2, self.M2))]

            self.Wx_o, self.Wc_o, self.Wh_o = [self.helper(dim) for dim in ((self.D, self.M2), 
									    (self.M2, self.M2), 
									    (self.M2, self.M2))]

        self.params = [self.Wx_i, self.Wc_i, self.Wh_i, self.Wx_f, self.Wc_f, self.Wh_f, self.Wx_c, self.Wh_c, self.Wx_o, self.Wc_o, self.Wh_o]

    def get_params(self):
        return self.params

        '''          =====================================          '''
        '''   ____________________________________________          '''
        '''  |                       |_____        |     |          '''
        '''  |                       ^     |       ^     |          '''
        ''' {X} => ||    cHat   ->   i ->  c ----> o --> h      ||  '''
        '''        ||      ^               ^ |     ^     |      ||  '''
        '''        ||      |         f -> /  |     |_____|      ||  '''
        '''        ||      |         ^       |           |      ||  '''
        '''        ||      |_________|_______|___________|      ||  '''
        '''i = sigm(x.dot(Wx_i) + h_rec(t-1).dot(Whrec_i) + c(t-1).dot(Wc_i) + bi) '''
        '''f = sigm(x.dot(Wx_f) + h_rec(t-1).dot(Whrec_f) + c(t-1).dot(Wc_f) + bf) '''
        '''c = f*c(t-1) + i*tanh(x.dot(Wx_c) + h_rec(t-1).dot(Whrec_c) + bc)'''
        '''o = sigm(x.dot(Wx_o) + h_rec(t-1).dot(Whrec_o) + c.dot(Wc_o) + bo)'''
        '''h_rec = o*tanh(c)'''

    def helper(self, dim):
        return list(map(tf.Variable, init_weight_and_bias(dim[0], dim[1])))

    def nonlinear(self, func):
        nonlin_dict = {"relu" : tf.nn.relu, 
		       "tanh" : tf.nn.tanh, 
		       "softmax" : tf.nn.softmax, 
		       "sigmoid" : tf.nn.sigmoid, 
		       "None" : lambda x: x}

        if func in nonlin_dict.keys():
            return nonlin_dict[func]

    def recurrence(self, prev_h_c, current_X):
        current_X = tf.reshape(current_X, (1, self.D))
        prev_h_rec = tf.reshape(prev_h_c[0], (1, self.M2))
        prev_c_rec = tf.reshape(prev_h_c[1], (1, self.M2))

        inp_g = self.nonlinear("sigmoid")(
					  tf.matmul(current_X, self.Wx_i[0]) + \
    					  tf.matmul(prev_h_rec, self.Wh_i[0]) + \ 
                                          tf.matmul(prev_c_rec, self.Wc_i[0]) + \
					  self.Wc_i[1])

        forget_g = self.nonlinear("sigmoid")(
                                             current_X[1] + \
                                             tf.matmul(prev_h_rec_, self.Wh_f[0]) + \
                                             tf.matmul(prev_c_rec_, self.Wc_f[0]) + \
                                             self.Wc_f[1])

        c = forget_g * prev_c_rec_ + inp_g * self.nonlinear("tanh")(
                                                                    current_X[2] + \
                                                                    tf.matmul(prev_h_rec_, self.Wh_c[0]) + \
                                                                    self.Wh_c[1])

        out = self.nonlinear("sigmoid")(
                                        current_X[3] + \
                                        tf.matmul(prev_h_rec_, self.Wh_o[0]) + \
                                        tf.matmul(c, self.Wc_o[0]) + \
                                        self.Wc_o[1])

        h_recur = out * self.nonlinear("tanh")(c)

        ''' return tuple (h(t-1), c(t-1)) because recurrence in tf may have only 2 variables '''
        return (tf.reshape(h_recur, (self.M2,)), tf.reshape(c, (self.M2,)))

    def output(self, Xw):
        h_hidden = tf.scan(fn=self.recurrence,  			# function to apply to each elems
                           elems=Xw,  					# fn takes each of element in elems
                           initializer=(self.Wh_c[1], self.Wc_o[1]), )
        return h_hidden[0]






class RecurrentPoetryClass():
    '''  M2 - num of recurrent hidden units layer, V - vocabulary size (num of distinct part of speech)'''
    def __init__(self, V, M2):
        self.V, self.M2 = V, M2
        self.reconstructed = False

    def nonlinear(self, func):
        nonlin_dict = {"relu" : tf.nn.relu, 
		       "tanh" : tf.nn.tanh, 
		       "softmax" : tf.nn.softmax, 
		       "sigmoid" : tf.nn.sigmoid, 
		       "None" : lambda x: x}

        if func in nonlin_dict.keys():
            return nonlin_dict[func]

    def set_session(self, session):
        self.session = session

    def optimizer(self, auto_optimizer, auto_opt_args):
        optimizer_dict = {"graddes": tf.train.GradientDescentOptimizer, 
			  "adadelta": tf.train.AdadeltaOptimizer, 
                          "adagrad": tf.train.AdagradOptimizer, 
			  "adagradD": tf.train.AdagradDAOptimizer, 
                          "momentum": tf.train.MomentumOptimizer, 
			  "adam": tf.train.AdamOptimizer, "ftlr": tf.train.FtrlOptimizer, 
                          "proxgrad": tf.train.ProximalGradientDescentOptimizer, 
			  "proxadagrad": tf.train.ProximalAdagradOptimizer, 
                          "rms": tf.train.RMSPropOptimizer}

        if auto_optimizer.lower() in optimizer_dict.keys():
            optimizer = optimizer_dict[auto_optimizer.lower()]
        else:
            raise ValueError('UNSUPPORTED OPTIMIZER TYPE')

        try:
            return optimizer(*auto_opt_args)
        except ValueError:
            print("Uncorrect arguments for " + optimizer + " optimizer")

    def helper(self, dim):
        return list(map(tf.Variable, init_weight_and_bias(dim[0], dim[1])))

    def model_initializer(self, recurr_unit, N_cls, nonlin_func, optimizer="adam", optimizer_args=(1e-5, 0.99, 0.999), reg=10e-3, 
                          train_mode=True, lst_W_b=None):

        self.tfX = tf.placeholder(tf.float32, shape=(None, self.V), name="tfX")
        self.tfT = tf.placeholder(tf.int32, shape=(None), name="tfT")

        if train_mode:
            #self.Wx_input, self.W_recurrent, self.W_output = [self.helper(dim) for dim in ((self.V, self.M2), 
	    #										    (self.M2, self.M2), 
            #                                                                               (self.M2, N_cls))]

            self.W_out = self.helper((self.M2, N_cls))
            self.RecUnit = recurr_unit(self.V, self.M2, nonlin_func)
        else:
            #self.Wx_input, self.W_recurrent, self.W_output = [list(map(tf.Variable, W_b)) for W_b in lst_W_b]
            self.W_out = lst_W_b[0]
            self.RecUnit = recurr_unit(self.V, self.M2, nonlin_func, Wb_npz=lst_W_b[1:])
            self.reconstructed = True

        self.params = [self.W_out] + self.RecUnit.get_params()

        h_hidden = self.RecUnit.output(self.tfX)

        # get output - return only the last element of sequence - we make prediction for whole poetry line, not for each next word
        self.logits = (tf.matmul(h_hidden, self.W_out[0]) + self.W_out[1])[-1]
        self.prediction = tf.argmax(tf.nn.softmax(self.logits))

        rcost = reg * sum([tf.nn.l2_loss(coefs) for weight_and_bias in self.params for coefs in weight_and_bias])
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(self.logits, (1, -1)), 
                                                                                  labels=tf.reshape(self.tfT, (1, )))) + rcost

        self.train_op = self.optimizer(optimizer, optimizer_args).minimize(self.cost)

    def fit(self, X, Y, recurr_unit, nonlin_func, optimizer, optimizer_args, reg, epochs=500, show_fig=False):

        N_cls = np.unique(Y).shape[0]
        self.model_initializer(recurr_unit, N_cls, nonlin_func, optimizer, optimizer_args, reg)

        self.session.run(tf.global_variables_initializer())
        final_correct_rate, final_epoch = 0, 0
        costs, corr_rate = [0]*epochs, [0]*epochs
        for epoch in range(epochs):
            X, Y = shuffle(X, Y)
            for sentense in range(len(X)):
                self.session.run(self.train_op, feed_dict={self.tfX: x_hot_encoding(X[sentense], self.V), self.tfT: Y[sentense]})
                c, predict = self.session.run([self.cost, self.prediction], feed_dict={self.tfX: x_hot_encoding(X[sentense], self.V), 
                                                                                       self.tfT: Y[sentense]})
                costs[epoch] += c

                corr_rate[epoch] += int(predict == Y[sentense])

            corr_rate[epoch] = corr_rate[epoch] / len(Y)
            print("epoch: ", epoch, " cost: ", costs[epoch], " correct rate: ", corr_rate[epoch])

            if epoch == epochs-1:
                final_correct_rate = corr_rate[-1]
                final_epoch = epochs

            if corr_rate[-1] >= 0.98:
                final_correct_rate = corr_rate[-1]
                final_epoch = epoch
                break

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.grid()
        ax.plot(np.log(costs))
        ax.set_title('Log cost Curve' + " corr_rate: " + str(final_correct_rate))
        if show_fig:
            plt.show()
        else:
            name_dict = {LSTM: "LSTM", 
			 GateReccUnit: "GateReccUnit", 
			 RateReccUnit: "RateReccUnit", 
			 SimpleReccUnit: "SimpleReccUnit"}

            name = "RNNclasif_" + name_dict[recurr_unit] + optimizer + "_".join(map(str, optimizer_args)) + str(final_epoch) + "_reg_" + \
                    str(reg) + ".png"
            fig.savefig(name)



    def predict(self, sentence):
        a = self.RecUnit.Wx_i[0]
        if self.reconstructed:
            self.session.run(tf.global_variables_initializer())
        assert((a.eval() == self.RecUnit.Wx_i[0].eval()).all())
        return self.session.run(self.prediction, feed_dict={self.tfX: x_hot_encoding(sentence, self.V)})

    #def score(self, sentence, target):
    #    p = self.predict(sentence)
    #    return 1 - np.mean(np.argmax(target, axis=1) != p)

    def save(self, filename):
        actual_params = self.session.run(self.params)
        np.savez(filename, *[p for W_and_b in actual_params for p in W_and_b])

    @staticmethod
    def load(filename, activation):
        ''' TODO: would prefer to save activation to file too '''
        npz = np.load(f(filename))
        param_num = len([elem for elem in npz])
        # [1, 2, 3, 4, 5, 6] => [(1, 2), (3, 4), (5, 6)]
        mylist = [npz['arr_' + str(n)] for n in range(param_num)]
        lst_W_b = list(zip(mylist[0::2], mylist[1::2]))

        M2, N_cls = npz['arr_0'].shape
        V, _ = npz['arr_2'].shape

        unit = {4: SimpleReccUnit, 
		8: RateReccUnit, 
		12: GateReccUnit, 
		22: LSTM}

        rnn = RecurrentPoetryClass(V, M2)
        # self.W_out (W and bias) = 2; num of recUnit params = length of npz - 2
        rnn.model_initializer(recurr_unit=unit[param_num - 2], N_cls=N_cls, nonlin_func=activation, train_mode=False, lst_W_b=lst_W_b)
        return rnn


''' general functions '''
def getAndSaveData(lst_poets, filename):
    # lst like ["poe.txt", "robert_frost.txt", "shakespeare.txt"]
    # filename like "poe_frost_shakespeare.npz"
    X, Y, word_id_dict = get_poetry_classifier_data(lst_poets, filename)
    return X, Y, word_id_dict

def getSavedData(filename):
    # filename like "poe_frost_shakespeare.npz"
    npz = np.load(f(filename))
    X, Y, word_id_dict = npz['arr_0'], npz['arr_1'], npz['arr_2']
    return X, Y, word_id_dict

def verifyModel(lst_poets, poetsfilename, hidden_units, recurr_unit, from_file):
    if from_file:
        npz = np.load(f(poetsfilename))
        X, Y, V = npz['arr_0'], npz['arr_1'], int(npz['arr_2'])
    else:
        X, Y, word_id_dict = getAndSaveData(lst_poets, poetsfilename)
        V = len(word_id_dict)

    X, Y = shuffle(X, Y)

    recurrent_model = RecurrentPoetryClass(V=V, M2=hidden_units)
    recurr_unit = recurr_unit
    session = tf.InteractiveSession()
    recurrent_model.set_session(session)

    N_verify = 250

    recurrent_model.fit(X[:-250], Y[:-250], recurr_unit, nonlin_func="relu", optimizer="adam", optimizer_args=(1e-4, 0.9, 0.999), reg=10e-10, 
                        epochs=150, show_fig=False)
    quality = 0
    for i in range(1, N_verify + 1):
        prediction = recurrent_model.predict(X[-i])
        label = Y[-i]
        if prediction == label:
            quality += 1
    print("quality: ", quality * 1.0 / N_verify)



def createModel(lst_poets, poetsfilename, hidden_units, modelfilename, recurr_unit, from_file):
    if from_file:
        npz = np.load(f(poetsfilename))
        X, Y, V = npz['arr_0'], npz['arr_1'], int(npz['arr_2'])
    else:
        X, Y, word_id_dict = getAndSaveData(lst_poets, poetsfilename)
        V = len(word_id_dict)
    X, Y = shuffle(X, Y)

    recurrent_model = RecurrentPoetryClass(V=V, M2=hidden_units)
    session = tf.InteractiveSession()
    recurrent_model.set_session(session)

    recurrent_model.fit(X, Y, recurr_unit, nonlin_func="relu", optimizer="adam", optimizer_args=(1e-4, 0.9, 0.999), reg=10e-10, epochs=150, 
                        show_fig=True)
    recurrent_model.save(modelfilename)


def loadModel(modelfilename, activation):
    rnn = RecurrentPoetryClass.load(modelfilename, activation)
    session = tf.InteractiveSession()
    rnn.set_session(session)
    return rnn





''' hidden_units=40 GOOD; two poets - 120 iterations enough; three poets - 200 '''

if __name__ == "__main__":
    
    # TWO POETS
    #getAndSaveData(["poe.txt", "robert_frost.txt"], "poe_frost.npz")
    ''' [LSTM, GateReccUnit, RateReccUnit, SimpleReccUnit] '''
    #for model in [LSTM, GateReccUnit, RateReccUnit, SimpleReccUnit]:
    #    verifyModel(["poe.txt", "robert_frost.txt"], "poe_frost.npz", 50, model, from_file=True)
    
    #createModel(["poe.txt", "robert_frost.txt"], "poe_frost.npz", 50, "LSTMPoeFrost_adam10-4_0.9_150ep.npz", LSTM, from_file=True)


    #LOAD PRETRAINED MODEL AND CHECK ITS QUALITY
    '''
    loaded_model = loadModel("LSTMPoeFrost_adam10-4_0.9_150ep.npz", "relu")
    X_test, Y_test, d = get_poetry_classifier_data(["poe_frost_verify.txt"], "poe_frost_verify.npz")
    X_test, Y_test = shuffle(X_test, Y_test)
    
    quality = 0
    for i in range(Y_test.shape[0]):
        if loaded_model.predict(X_test[i]) == Y_test[i]:
            quality += 1
    print("quality: ", quality * 1.0 / Y_test.shape[0])
    '''

    # THREE POETS
    #getAndSaveData(["poe.txt", "shakespeare.txt", "robert_frost.txt"], "poe_shakespeare_frost.npz")
    #for model in [RateReccUnit, SimpleReccUnit]:
    #    verifyModel(["poe.txt", "shakespeare.txt", "robert_frost.txt"], "poe_shakespeare_frost.npz", 50, model, from_file=True)
    #createModel(["poe.txt", "shakespeare.txt", "robert_frost.txt"], "poe_shakespeare_frost.npz", 50, 
    #             "LSTMPoeShakesFrost_adam10-4_0.9_150ep.npz", LSTM, from_file=True)

    '''
    X_test, Y_test, d = get_poetry_classifier_data(["poe_shaks_frost_verify.txt"], "poe_shakespeare_frost_verify.npz")
    X_test, Y_test = shuffle(X_test, Y_test)

    quality = 0
    for i in range(Y_test.shape[0]):
        if loaded_model.predict(X_test[i]) == Y_test[i]:
            quality += 1
    print("quality: ", quality * 1.0 / Y_test.shape[0])
    '''


    # USE ONLY TRAIN DATA
    '''
    N_verify = 150

    X, Y = shuffle(X, Y)

    recurrent_model = RecurrentPoetryClass(V=len(word_id_dict), M2=40)
    session = tf.InteractiveSession()
    recurrent_model.set_session(session)
    for i in range(1):
        X, Y = shuffle(X, Y)
        recurrent_model.fit(X[:-150], Y[:-150], nonlin_func="relu", optimizer="adam", optimizer_args=(1e-4, 0.99, 0.999), reg=10e-10, 
                            epochs=220, show_fig=True)
        quality = 0
        for i in range(1, N_verify + 1):
            prediction = recurrent_model.predict(X[-i])
            label = Y[-i]
            if prediction == label:
                quality += 1
            print("prediction: ", prediction, " true label: ", label)
        print("quality: ", quality * 1.0 / N_verify)
    '''
    

    
    '''
    for k in range(8):
        recurrent_model.fit(X=(X[:k * sz] + X[(k * sz + sz):]), Y=(Y[:k * sz] + Y[(k * sz + sz):]), nonlin_func="relu", optimizer="adam", 
                            optimizer_args=(1e-4, 0.99, 0.999), reg=0, epochs=100, show_fig=False)
        err = []
        for index in range(k * sz, (k * sz + sz)):
            err.append(int(recurrent_model.predict(X[index]) == Y[index]))
        errors.append(sum(err) * 1.0 / len(err))
    print(np.mean(np.array(errors)))
    '''