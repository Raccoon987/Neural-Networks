import tensorflow as tf
import numpy as np



def init_weight_and_bias(M1, M2):
    return (np.random.randn(M1, M2) * np.sqrt(2.0 / M1)).astype(np.float32), (np.zeros(M2)).astype(np.float32)

    # return (np.random.randn(M1, M2) / np.sqrt(M2 + M1)).astype(np.float32), (np.zeros(M2)).astype(np.float32)
    # return (np.random.uniform(low=-np.sqrt(6/(M2 + M1)), high=np.sqrt(6/(M2 + M1)), size=(M1, M2))).astype(np.float32), \
    #         (np.zeros(M2)).astype(np.float32) #sigmoid
    # return (np.random.uniform(low=-4*np.sqrt(6/(M2 + M1)), high=4*np.sqrt(6/(M2 + M1)), size=(M1, M2))).astype(np.float32), \
    #         (np.zeros(M2)).astype(np.float32) #tanh



class SimpleReccUnit():
    def __init__(self, D, M2, nonlin_func, Wb_npz=None):
        self.D, self.M2 = D, M2
        self.nonlin_func = nonlin_func
        self.set_Wb(Wb_npz)

    def set_Wb(self, Wb_npz):
        if Wb_npz:
            self.Wx_h, self.Wh_h, self.Wx_0 = [list(map(tf.Variable, W_b)) for W_b in Wb_npz]
        else:
            self.Wx_h, self.Wh_h = [self.helper(dim) for dim in ((self.D, self.M2),
                                                                 (self.M2, self.M2))]
            #self.Wx_0 = self.helper((self.M2, self.M2)) #use only second bias term to initialize first "current_X" value of recurrence

        self.params = [self.Wx_h, self.Wh_h]         #, self.Wx_0]

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


    def get_hidden_value(self, prev_h_rec, current_X):
        current_X = tf.reshape(current_X, (1, self.M2))
        prev_h_rec = tf.reshape(prev_h_rec, (1, self.M2))

        h_recur = self.nonlinear(self.nonlin_func)(
                                                   current_X + \
                                                   tf.matmul(tf.reshape(prev_h_rec, (1, self.M2)), self.Wh_h[0]) + \
                                                   self.Wh_h[1])

        return tf.reshape(h_recur, (self.M2,))


    def recurrence(self, prev_h_rec, current_X):
        result_h = tf.cond(
                           pred=tf.equal(current_X[1], tf.constant(1)),
                           true_fn=lambda: self.get_hidden_value(self.Wh_h[1], current_X[0]),
                           false_fn=lambda: self.get_hidden_value(prev_h_rec, current_X[0]))

                  #tf.cond(
                  #        pred=tf.equal(current_X[1], tf.constant(1)),
                  #        true_fn=lambda: self.get_hidden_value(self.Wx_0[1], current_X[0]),
                  #        false_fn=lambda: self.get_hidden_value(prev_h_rec, current_X[0]))

        return result_h

    def output(self, Xw, startPoints):
        x_h = tf.matmul(Xw, self.Wx_h[0])

        h_hidden = tf.scan(fn=self.recurrence,  # function to apply to each elems
                           elems=(x_h, startPoints),  # fn takes each of element in elems
                           initializer=self.Wh_h[1], )              #self.Wx_0[1], )

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

    def get_hidden_value(self, prev_h_rec, current_X):
        current_X = list(map(lambda x, y: tf.reshape(x, y), current_X, [(1, self.M2)] * 2))
        prev_h_rec = tf.reshape(prev_h_rec, (1, self.M2))

        hHat_recur = self.nonlinear(self.nonlin_func)(
                                                      current_X[0] + \
                                                      tf.matmul(prev_h_rec, self.Wh_h[0]) + \
                                                      self.Wh_h[1])

        z_gate = self.nonlinear("sigmoid")(
                                           current_X[1] + \
                                           tf.matmul(prev_h_rec, self.Wh_z[0]) + \
                                           self.Wh_z[1])

        h_recur = (1 - z_gate) * prev_h_rec + z_gate * hHat_recur
        return tf.reshape(h_recur, (self.M2,))

    def recurrence(self, prev_h_rec, current_X):
        result_h = tf.cond(
                           pred=tf.equal(current_X[1], tf.constant(1)),
                           true_fn=lambda: self.get_hidden_value(self.Wx_h[1], current_X[0]),
                           false_fn=lambda: self.get_hidden_value(prev_h_rec, current_X[0]))
        return result_h

    def output(self, Xw, startPoints):
        elems = list(map(lambda x, y: tf.matmul(x, y) [Xw]*2, [self.Wx_h[0], self.Wx_z[0]]))

        h_hidden = tf.scan(fn=self.recurrence,  # function to apply to each elems
                           elems=(elems, startPoints),  # fn takes each of element in elems
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

    def get_hidden_value(self, prev_h_rec, current_X):
        current_X = list(map(lambda x, y: tf.reshape(x, y), current_X, [(1, self.M2)] * 3))
        prev_h_rec = tf.reshape(prev_h_rec, (1, self.M2))

        r = self.nonlinear("sigmoid")(
                                      current_X[0] + \
                                      tf.matmul(prev_h_rec, self.Wh_r[0]) + \
                                      self.Wh_r[1])

        z_gate = self.nonlinear("sigmoid")(
                                           current_X[1] + \
                                           tf.matmul(prev_h_rec, self.Wh_z[0]) + \
                                           self.Wh_z[1])

        hHat_recur = self.nonlinear(self.nonlin_func)(
                                                      current_X[2] + \
                                                      r * tf.matmul(prev_h_rec, self.Wh_h[0]) + \
                                                      self.Wh_h[1])

        h_recur = (1 - z_gate) * prev_h_rec + z_gate * hHat_recur
        return tf.reshape(h_recur, (self.M2,))

    def recurrence(self, prev_h_rec, current_X):
        result_h = tf.cond(
                           pred=tf.equal(current_X[1], tf.constant(1)),
                           true_fn=lambda: self.get_hidden_value(self.Wx_h[1], current_X[0]),
                           false_fn=lambda: self.get_hidden_value(prev_h_rec, current_X[0]))
        return result_h

    def output(self, Xw, startPoints):
        elems = list(map(lambda x, y: tf.matmul(x, y), [Xw]*3, [self.Wx_r[0], self.Wx_z[0], self.Wx_h[0]]))

        h_hidden = tf.scan(fn=self.recurrence,  # function to apply to each elems
                           elems=(elems, startPoints),  # fn takes each of element in elems
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

            #self.Wh_0, self.Wc_0 = [self.helper(dim) for dim in ((self.M2, self.M2),
            #                                                     (self.M2, self.M2))]



        self.params = [self.Wx_i, self.Wc_i, self.Wh_i, self.Wx_f, self.Wc_f, self.Wh_f, self.Wx_c, self.Wh_c, self.Wx_o, self.Wc_o, self.Wh_o]#, self.Wh_0, self.Wc_0]

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


    def get_hidden_value(self, prev_h_rec, current_X):
        current_X = list(map(lambda x, y: tf.reshape(x, y), current_X, [(1, self.M2)] * 4))

        prev_h_rec_ = tf.reshape(prev_h_rec[0], (1, self.M2))
        prev_c_rec_ = tf.reshape(prev_h_rec[1], (1, self.M2))

        inp_g = self.nonlinear("sigmoid")(
                                          current_X[0] + \
                                          tf.matmul(prev_h_rec_, self.Wh_i[0]) + \
                                          tf.matmul(prev_c_rec_, self.Wc_i[0]) + \
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

    def recurrence(self, prev_h_rec, current_X):
        result_h = tf.cond(
                           pred=tf.equal(current_X[1], tf.constant(1)),
                           true_fn=lambda: self.get_hidden_value((self.Wh_c[1], self.Wc_o[1]), current_X[0]),
                           false_fn=lambda: self.get_hidden_value(prev_h_rec, current_X[0]))
        return result_h

    def output(self, Xw, startPoints):
        elems = list(map(lambda x, y: tf.matmul(x, y), [Xw]*4, [self.Wx_i[0], self.Wx_f[0], self.Wx_c[0], self.Wx_o[0]]))

        h_hidden = tf.scan(fn=self.recurrence,          # function to apply to each elems
                           elems=(elems, startPoints),  # !!! CURRENT_X !!! fn takes each of element in elems = [Xw.dot(Wx_i[0]), Xw.dot(Wx_f[0]), Xw.dot(Wx_c[0]), Xw.dot(Wx_o[0])]
                           initializer=(self.Wh_c[1], self.Wc_o[1]), )
                           #initializer=(self.Wh_0[1], self.Wc_0[1]), )
        return h_hidden[0]