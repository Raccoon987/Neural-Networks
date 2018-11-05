import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import datetime
import sys
import operator
from sklearn.utils import shuffle
import sklearn.model_selection
from sklearn.metrics import f1_score


f_path = lambda f_name: os.path.realpath(os.path.join(os.getcwd(), f_name)).replace('\\', '/')

def init_weight_and_bias(args):
    #return (np.random.randn(*args) * np.sqrt(2.0 / args[-1])).astype(np.float32), (np.zeros(args[-1])).astype(np.float32)
    return (np.random.randn(*args) / np.sqrt(sum(args))).astype(np.float32), (np.zeros(args[-1])).astype(np.float32)


class Stack:
    def __init__(self):
        self.items = []
        self.inserts_counter = 0

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)
        self.inserts_counter += 1

    def pop(self):
        if self.items:
            return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)

    def get_counter(self):
        return self.inserts_counter


class BinTreeNode:
    def __init__(self, word, label, index):
        self.idx = index
        self.leftChild = None
        self.rightChild = None
        self.word = word
        self.label = label

    def insertLeft(self, tree_node):
        self.leftChild = tree_node

    def insertRight(self, tree_node):
        self.rightChild = tree_node

    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def get_word(self):
        return self.word

    def get_label(self):
        return self.label

    def get_idx(self):
        return self.idx

    def set_idx(self, num):
        self.idx = num


def buildParseTree(string, word2idx):
    string = string.replace(" ", "")
    pStack = Stack()
    t = BinTreeNode(word=-1, label=string[1], index=0)
    pStack.push(t)

    for idx, element in enumerate(string[2:]):
        if element == "(":
            sub_one = string[2:][idx+1:].rsplit("(")[0]
            sub_two = string[2:][idx+1:].rsplit(")")[0]
            compare = lambda x: len(x[0]) < len(x[1])

            if compare((sub_one, sub_two)):
                node = BinTreeNode(word=-1, label=sub_one, index=(pStack.get_counter())+1)
            else:
                word2idx[sub_two[1:]] = word2idx.get(sub_two[1:], 1+len(word2idx))
                node = BinTreeNode(word=word2idx[sub_two[1:]], label=sub_two[:1], index=(pStack.get_counter())+1)

            parentNode = pStack.peek()
            if parentNode != None:
                if parentNode.getLeftChild() == None:
                    parentNode.insertLeft(node)
                else:
                    parentNode.insertRight(node)
            pStack.push(node)

        elif element == ")":
            pStack.pop()

    global counter
    counter = 0
    def postorder_idx(tree):
        # post-order labeling of tree nodes
        global counter
        if tree:
            postorder_idx(tree.getLeftChild())
            postorder_idx(tree.getRightChild())
            tree.set_idx(counter)
            counter += 1

    postorder_idx(t)
    return t


def preorder(tree):
    if tree is not None:
        if tree.get_word() != -1:
            print(tree.get_word(), end="  ")
            print(tree.get_idx(), [tree.getLeftChild().get_idx() if tree.getLeftChild() is not None else -1], [tree.getRightChild().get_idx() if tree.getRightChild() is not None else -1])
        preorder(tree.getLeftChild())
        preorder(tree.getRightChild())


def postorder(tree):
    if tree is not None:
        preorder(tree.getLeftChild())
        preorder(tree.getRightChild())
        if tree.get_word() != -1:
            print(tree.get_word(), end="  ")
            print(tree.get_idx(), [tree.getLeftChild().get_idx() if tree.getLeftChild() is not None else -1], [tree.getRightChild().get_idx() if tree.getRightChild() is not None else -1])


def get_tree_lst(tree_node, parent_id):
    if tree_node is None:
        return [], [], [], []

    ''' get word, left, right child id and label of the current node left and right childrens  '''
    leftChildWord, leftLeftCh_id, leftRightCh_id, leftChildLabel = get_tree_lst(tree_node.getLeftChild(), tree_node.get_idx())
    rightChildWord, rightLeftCh_id, rightRightCh_id, rightChildLabel = get_tree_lst(tree_node.getRightChild(), tree_node.get_idx())

    wordsList = leftChildWord + rightChildWord + [tree_node.get_word()]
    leftChList = leftLeftCh_id + rightLeftCh_id + [tree_node.getLeftChild().get_idx() if (tree_node.get_word() == -1) else -1]
    rightChList = leftRightCh_id + rightRightCh_id + [tree_node.getRightChild().get_idx() if (tree_node.get_word() == -1) else -1]
    labelsList = leftChildLabel + rightChildLabel + [tree_node.get_label()]

    return wordsList, leftChList, rightChList, list(map(int, labelsList))





def get_data(train_filename, test_filename):

    train_trees, test_trees, word2idx = [], [], {}

    f = lambda args: [args[1].append(buildParseTree(line.rstrip(), word2idx)) if line.rstrip() else None for line in open(args[0])]

    try:
        f((train_filename, train_trees))
        f((test_filename, test_trees))

        return train_trees, test_trees, word2idx

    except IOError:
        print("file not exist")


class TNN():
    def __init__(self, D):
        self.D = D

    def nonlinear(self, func):
        nonlin_dict = {"relu": tf.nn.relu,
                       "tanh": tf.nn.tanh,
                       "softmax": tf.nn.softmax,
                       "sigmoid": tf.nn.sigmoid,
                       "None": lambda x: x}
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
                          "adam": tf.train.AdamOptimizer,
                          "ftlr": tf.train.FtrlOptimizer,
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
        return list(map(tf.Variable, init_weight_and_bias(dim)))


    def get_labels(self, tree):
        if tree is None :
            return []
        return self.get_labels(tree.getLeftChild()) + self.get_labels(tree.getRightChild()) + [tree.get_label()]


    def dotVecMat(self, v, M):
        return tf.tensordot(v, M, axes=[[0], [1]])


    def dotMatVec(self, M, v):
        return tf.tensordot(M, v, axes=[[1], [0]])


    def forward(self, hid_iter, n):
        left = hid_iter.read(tf.gather(self.leftChild, n))                     # get node's left child value
        right = hid_iter.read(tf.gather(self.rightChild, n))                   # get node's right child value

        ''' self.W_1[0] - weight, self.W_1[1] - bias'''
        #left = tf.Print(left, ["left shape: ", tf.shape(left), "self.W_11 shape: ", tf.shape(self.W_11)])
        return self.nonlinear(self.f)(self.dotVecMat(left, self.dotMatVec(self.W_11, left)) + \
                                      self.dotVecMat(right, self.dotMatVec(self.W_22, right)) + \
                                      self.dotVecMat(left, self.dotMatVec(self.W_12, right)) + \
                                      self.dotVecMat(left, self.W_1[0]) + \
                                      self.dotVecMat(right, self.W_2) + \
                                      self.W_1[1])



    def body(self, hid_iter, n):
        w = tf.gather(self.words, n)                                             # get current word index

        hidden_value = tf.cond(
                               w >= 0,
                               lambda: tf.nn.embedding_lookup(self.W_embed, w),  # select word vector or...
                               lambda: self.forward(hid_iter, n)                 # calculate hidden value based on left, right childs value
        )

        hid_iter = hid_iter.write(n, hidden_value)                               # write new value to TensorArray
        n = tf.add(n, 1)                                                         # increment step

        return hid_iter, n


    def model_initializer(self, V, K, nonlin_func, optimizer="adam", optimizer_args=(1e-5, 0.99, 0.999), reg=1e-2, train_inner_nodes=False, load_w=None):

        self.V = V
        self.K = K
        self.f = nonlin_func

        self.words = tf.placeholder(tf.int32, shape=(None,), name="words")
        self.leftChild = tf.placeholder(tf.int32, shape=(None,), name="leftChd")
        self.rightChild = tf.placeholder(tf.int32, shape=(None,), name="rightChd")
        self.labels = tf.placeholder(tf.int32, shape=(None,), name="labels")

        if load_w:
            self.W_embed = tf.Variable(load_w["arr_0"])
            self.W_11 = tf.Variable(load_w["arr_1"])
            self.W_12 = tf.Variable(load_w["arr_2"])
            self.W_22 = tf.Variable(load_w["arr_3"])
            self.W_1 = [tf.Variable(load_w["arr_4"]), tf.Variable(load_w["arr_5"])]
            self.W_2 = tf.Variable(load_w["arr_6"])
            self.W_out = [tf.Variable(load_w["arr_7"]), tf.Variable(load_w["arr_8"])]
        else:
            self.W_embed = self.helper((self.V, self.D))[0]                             # only weights, no bias
            self.W_11 = self.helper((self.D, self.D, self.D))[0]                        # only weights, no bias
            self.W_12 = self.helper((self.D, self.D, self.D))[0]                        # only weights, no bias
            self.W_22 = self.helper((self.D, self.D, self.D))[0]                        # only weights, no bias
            self.W_1 = self.helper((self.D, self.D))
            self.W_2 = self.helper((self.D, self.D))[0]                                 # only weights, no bias
            self.W_out = self.helper((self.D, self.K))

        self.params = [self.W_embed, self.W_11, self.W_12, self.W_22, self.W_1, self.W_2, self.W_out]

        def condition(hid_iter, n):
            return tf.less(n, tf.shape(self.words)[0])

        hid_output, _ = tf.while_loop(
                                      cond = condition,                                             # when to stop
                                      body = self.body,                                             # what to do
                                      loop_vars = [tf.TensorArray(dtype=tf.float32,                 # init values
                                                                  size=0,
                                                                  dynamic_size=True,
                                                                  clear_after_read=False,
                                                                  infer_shape=False),
                                                   tf.constant(0, dtype=tf.int32)],
                                      parallel_iterations = 1
        )

        print(hid_output)
        hid_output= hid_output.stack()
        print(hid_output)
        logits = tf.matmul(hid_output, self.W_out[0]) + self.W_out[1]
        self.prediction = tf.argmax(logits, axis=1)

        ''' sum only weights '''
        rcost = reg * sum([tf.nn.l2_loss(w) for w in [self.W_embed, self.W_11, self.W_12, self.W_22, self.W_2]])
        ''' sum weights and biases '''
        rcost += reg * sum([tf.nn.l2_loss(w_b) for w_and_b in [self.W_1, self.W_out] for w_b in w_and_b])

        if train_inner_nodes:
            idx = tf.where(self.labels >= 0)
            logits_ = tf.gather(logits, idx)
            labels_ = tf.gather(self.labels, idx)
        else:
            logits_ = logits[-1]
            labels_ = self.labels[-1]

        labels_ = tf.Print(labels_, ["logits_: ", logits_, "logits_ shape: ", tf.shape(logits_), "labels_: ", labels_,
                                     "labels_ shape: ", tf.shape(labels_), "self.prediction shape: ", tf.shape(self.prediction), "idx: ", idx, "idx shape: ", tf.shape(idx)])

        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits_,
                labels=labels_)) + rcost

        self.train_op = self.optimizer(optimizer, optimizer_args).minimize(self.cost)


    def fit(self, tree_lsts, V, K, nonlin_func, optimizer, optimizer_args, epochs, reg, train_inner_nodes):

        train_set, valid_set = sklearn.model_selection.train_test_split(np.array(tree_lsts), test_size=0.15, random_state=42)

        self.model_initializer(V, K, nonlin_func, optimizer, optimizer_args, reg, train_inner_nodes)

        self.session.run(tf.global_variables_initializer())

        costs = [0] * epochs
        train_accuracy = [0] * epochs
        valid_accuracy = [0] * epochs

        for epoch in range(epochs):
            t0 = datetime.datetime.now()

            train_correct, valid_correct = [], []
            ''' train process '''
            for idx in shuffle(range(len(train_set))):
                words, leftCh, rightCh, labels = train_set[idx]

                self.session.run(self.train_op, feed_dict={self.words:words,
                                                           self.leftChild:leftCh,
                                                           self.rightChild:rightCh,
                                                           self.labels:labels})

                c, predict = self.session.run([self.cost, self.prediction], feed_dict={self.words:words,
                                                                                       self.leftChild:leftCh,
                                                                                       self.rightChild:rightCh,
                                                                                       self.labels:labels})
                costs[epoch] += c
                train_correct.append(predict[-1] == labels[-1])

                if idx % 1000 == 0:
                    sys.stdout.write("1000 sentences ")
                    sys.stdout.flush()

            train_accuracy[epoch] = round(sum(train_correct) / float(len(train_correct)), 3)

            """ validation process """
            for idx in shuffle(range(len(valid_set))):
                words, leftCh, rightCh, labels = valid_set[idx]

                predict = self.session.run(self.prediction, feed_dict={self.words: words,
                                                                         self.leftChild: leftCh,
                                                                         self.rightChild: rightCh,
                                                                         self.labels: labels})
                valid_correct.append(predict[-1] == labels[-1])

            valid_accuracy[epoch] = round(sum(valid_correct) / float(len(valid_correct)), 3)


            print("epoch: ", epoch, \
                  " cost: ", costs[epoch], \
                  "train correct rate: ", train_accuracy[epoch], \
                  "validation correct rate: ", valid_accuracy[epoch], \
                  " time for epoch: ", (datetime.datetime.now() - t0))

        fig, ax1 = plt.subplots()
        ax1.plot(np.log(costs), 'b-')
        ax1.set_xlabel('epochs')
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel('log cost', color='b')

        ax2 = ax1.twinx()
        ax2.plot(train_accuracy, 'r-')
        ax2.plot(valid_accuracy, 'g-')
        ax2.set_ylabel('train_accuracy', color='r')

        fig.tight_layout()
        ax1.set_title('Log cost Curve' + "train_acc: " + str(round(train_accuracy[-1], 3)) + \
                                         "valid_acc: " + str(round(valid_accuracy[-1], 3)))
        ax1.grid()
        plt.show()


    def save_weights(self, filename):
        ''' save all weights, embedding size and nonlinear_function'''
        actual_params = self.session.run(self.params)
        saved_lst = []
        for Wb in actual_params:
            if isinstance(Wb, list):
                for p in Wb:
                    saved_lst.append(p)
            else:
                saved_lst.append(Wb)
        saved_lst.extend([self.D, self.f, self.V, self.K])
        np.savez(filename, *saved_lst)


    @staticmethod
    def load_weights(filename):
        npz = np.load(filename)
        ''' W_embed, W_11, W_12, W_22, W_1, W_2, W_out, D, f_nonlin, V, K '''
        D, f_nonlin, V, K = int(npz["arr_9"]), str(npz["arr_10"]), int(npz["arr_11"]), int(npz["arr_12"])
        #print("D: ", type(int(D)), int(D))
        tnn = model = TNN(D)
        tnn.model_initializer(V, K, f_nonlin, load_w=npz)

        return tnn


    def predict(self, tree, k=False):
        ''' return root node label prediction '''
        words, leftCh, rightCh, labels = tree
        if k:
            self.session.run(tf.global_variables_initializer())
        return self.session.run(self.prediction, feed_dict={self.words: words,
                                                                         self.leftChild: leftCh,
                                                                         self.rightChild: rightCh,
                                                                         self.labels: labels})[-1]

    def score_prediction(self, p, l):
        """ return accuracy and f_1 score """
        return np.sum((np.equal(p, l))) / (1.0 * len(l)), f1_score(p, l, average=None).mean()









if __name__ == "__main__":

    '''
    word2idx = {}
    a = "(3 (2 It)(4 (4 (2 's) (4 (3 (2 a) (4 (3 lovely) (2 film))) (3 (2 with) (4 (3 (3 lovely) (2 performances)) (2 (2 by) (2 (2 (2 Buy) (2 and)) (2 Accorsi))))))) (2 .))) "
    b = "(1 (2 An) (1 (1 absurdist) (1 (2 spider) (2 (2 web) (2 .)))))"
    preorder(buildParseTree(a, word2idx))
    print(word2idx)
    '''

    train_trees, test_trees, word2idx = get_data("recursive_train.txt", "recursive_test.txt")
    index2word = {index: word for word, index in word2idx.items()}
    

    f_lst = lambda tree: [get_tree_lst(t, -1) for t in tree]
    train_lst = f_lst(train_trees)
    test_lst = f_lst(test_trees)

    V = len(word2idx)
    K = 5

    model = TNN(10)

    session = tf.InteractiveSession()
    model.set_session(session)
    model.fit(train_lst[:1200], V, K, nonlin_func="tanh", optimizer="adagrad", optimizer_args=(8e-3, ), epochs=30, reg=1e-3, train_inner_nodes=True)
    model.save_weights("recursiveNN_weights.npz")

    for i in range(30):
        print(model.predict(test_lst[i]), end=" ")

    print("test accuracy: %f test f1 score: %f", (model.score_prediction([model.predict(t) for t in test_lst[:500]],
                                                                         list(map(operator.itemgetter(-1), np.array(test_lst)[:500, 3]))
                                                                         )
                                                  ))


    model_1 = TNN.load_weights("recursiveNN_weights.npz")
    session = tf.InteractiveSession()
    model_1.set_session(session)

    for i in range(30):
        print(model_1.predict(test_lst[i], k=True), end=" ")

    print("test accuracy: %f test f1 score: %f", (model_1.score_prediction([model_1.predict(t) for t in test_lst[:500]],
                                                                           list(map(operator.itemgetter(-1), np.array(test_lst)[:500, 3]))
                                                                           )
                                                  ))



