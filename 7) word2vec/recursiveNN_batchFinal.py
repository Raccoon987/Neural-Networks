import tensorflow as tf
from tensorflow.python.util import nest
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
    # return (np.random.randn(*args) * np.sqrt(2.0 / args[-1])).astype(np.float32), (np.zeros(args[-1])).astype(np.float32)
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
    ''' build parse tree that consists from BinTreeNode-s;
        each node has idx that starts from 1 and is assigned in postorder tree traversal order;
        1 and not 0 as a start index was used because batch keras padding add zeros at the beginning of all sentences in
        batch that are shorter than the longest ones. And Nodes indexes in right and leftChild lists are used as a position
        of appropriate element in TensorArray;

        string looks like:
        (3 (2 It)(4 (4 (2 's) (4 (3 (2 a) (4 (3 lovely) (2 film))) (3 (2 with) (4 (3 (3 lovely) (2 performances))
        (2 (2 by) (2 (2 (2 Buy) (2 and)) (2 Accorsi))))))) (2 .)))

        word2idx: dictionary; key - unique word, value - unique index

        returns binary tree object
        '''
    string = string.replace(" ", "")
    pStack = Stack()
    ''' add +1 to label to account label = 0 after batch keras padding; '''
    t = BinTreeNode(word=-1,
                    label=int(string[1]) + 1,
                    index=0)
    pStack.push(t)

    for idx, element in enumerate(string[2:]):
        if element == "(":
            sub_one = string[2:][idx + 1:].rsplit("(")[0]
            sub_two = string[2:][idx + 1:].rsplit(")")[0]
            compare = lambda x: len(x[0]) < len(x[1])

            if compare((sub_one, sub_two)):
                ''' add +1 to label to account label = 0 after batch keras padding; '''
                node = BinTreeNode(word=-1,
                                   label=int(sub_one) + 1,
                                   index=(pStack.get_counter()) + 1)
            else:
                word2idx[sub_two[1:]] = word2idx.get(sub_two[1:], 1 + len(word2idx))
                node = BinTreeNode(word=word2idx[sub_two[1:]],
                                   label=int(sub_two[:1]) + 1,
                                   index=(pStack.get_counter()) + 1)

            parentNode = pStack.peek()
            if parentNode != None:
                if parentNode.getLeftChild() == None:
                    parentNode.insertLeft(node)
                else:
                    parentNode.insertRight(node)
            pStack.push(node)

        elif element == ")":
            pStack.pop()

    '''  index start not from 0 but from 1 because 0 means "empty keras padding. later do "-1" operation '''
    global counter
    counter = 1

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
            print(tree.get_idx(), [tree.getLeftChild().get_idx() if tree.getLeftChild() is not None else -1],
                  [tree.getRightChild().get_idx() if tree.getRightChild() is not None else -1])
        preorder(tree.getLeftChild())
        preorder(tree.getRightChild())


def postorder(tree):
    if tree is not None:
        postorder(tree.getLeftChild())
        postorder(tree.getRightChild())
        if tree.get_word() != -1:
            print(tree.get_word(), end="  ")
            print(tree.get_idx(), [tree.getLeftChild().get_idx() if tree.getLeftChild() is not None else -1],
                  [tree.getRightChild().get_idx() if tree.getRightChild() is not None else -1])


def get_tree_lst(tree_node, parent_id):
    ''' postorder tree traversal to form four lists;
        tree_node - tree - output of buildParseTree function;
        parent_id = -1;

        returns four lists - words (with unique word numnbers or -1 if node has no word),
        lists of left and right childrens of a node (idx of node or "-1" if node has no children),
        labels of each node '''

    if tree_node is None:
        return [], [], [], []

    ''' get word, left, right child id and label of the current node left and right childrens  '''
    leftChildWord, leftLeftCh_id, leftRightCh_id, leftChildLabel = get_tree_lst(tree_node.getLeftChild(),
                                                                                tree_node.get_idx())
    rightChildWord, rightLeftCh_id, rightRightCh_id, rightChildLabel = get_tree_lst(tree_node.getRightChild(),
                                                                                    tree_node.get_idx())

    wordsList = leftChildWord + rightChildWord + [tree_node.get_word()]

    leftChList = leftLeftCh_id + \
                 rightLeftCh_id + \
                 [tree_node.getLeftChild().get_idx() if (tree_node.get_word() == -1) else -1]

    rightChList = leftRightCh_id + \
                  rightRightCh_id + \
                  [tree_node.getRightChild().get_idx() if (tree_node.get_word() == -1) else -1]

    labelsList = leftChildLabel + rightChildLabel + [tree_node.get_label()]

    return wordsList, leftChList, rightChList, labelsList  # list(map(int, labelsList))


def get_data(train_filename, test_filename):
    '''  train_filename, test_filename - txt files with lines like:
         (3 (2 It)(4 (4 (2 's) (4 (3 (2 a) (4 (3 lovely) (2 film))) (3 (2 with) (4 (3 (3 lovely) (2 performances))
         (2 (2 by) (2 (2 (2 Buy) (2 and)) (2 Accorsi))))))) (2 .)))

         returns two lists of binary trees and dictionary unique_word-> unique index'''

    train_trees, test_trees, word2idx = [], [], {}

    f = lambda args: [args[1].append(buildParseTree(line.rstrip(), word2idx)) if line.rstrip() else None for line in
                      open(args[0])]

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
        if tree is None:
            return []
        return self.get_labels(tree.getLeftChild()) + self.get_labels(tree.getRightChild()) + [tree.get_label()]


    def dotVecMat(self, v, M):
        return tf.tensordot(v, M, axes=[[1], [1]])


    def nonlinear_mul(self, a_, M, _a):
        ''' multiply a batch of matrices with a batch of vectors of the same length, pairwise
            M shape (batch_size, n, m), v shape (batch_size, m) => Mv = (M @ v[..., None])[..., 0] shape (batch_size, n) '''
        return tf.matmul(
            tf.tensordot(_a, M, axes=[[1], [1]]),
            a_[..., None]
        )[..., 0]


    def slicing_where(self, condition, full_input, true_branch, false_branch):
        """Split `full_input` between `true_branch` and `false_branch` on `condition`.

        Args:
          condition: A boolean Tensor with shape [B_1, ..., B_N].
          full_input: A Tensor or nested tuple of Tensors of any dtype, each with shape [B_1, ..., B_N, ...], to be split
            between `true_branch` and `false_branch` based on `condition`.
          true_branch: A function taking a single argument, that argument having the same structure and number of batch
            dimensions as `full_input`. Receives slices of `full_input` corresponding to the True entries of `condition`.
            Returns a Tensor or nested tuple of Tensors, each with batch dimensions matching its inputs.
          false_branch: Like `true_branch`, but receives inputs corresponding to the false elements of `condition`. Returns
            a Tensor or nested tuple of Tensors (with the same structure as the return value of `true_branch`), but with
            batch dimensions matching its inputs.
        Returns:
          Interleaved outputs from `true_branch` and `false_branch`, each Tensor having shape [B_1, ..., B_N, ...].
        """
        full_input_flat = nest.flatten(full_input)
        true_indices = tf.where(condition)
        false_indices = tf.where(tf.logical_not(condition))
        true_branch_inputs = nest.pack_sequence_as(
            structure=full_input,
            flat_sequence=[tf.gather_nd(params=input_tensor, indices=true_indices) for input_tensor in full_input_flat])
        false_branch_inputs = nest.pack_sequence_as(
            structure=full_input,
            flat_sequence=[tf.gather_nd(params=input_tensor, indices=false_indices) for input_tensor in
                           full_input_flat])
        true_outputs = true_branch(true_branch_inputs)
        false_outputs = false_branch(false_branch_inputs)
        nest.assert_same_structure(true_outputs, false_outputs)

        def scatter_outputs(true_output, false_output):
            batch_shape = tf.shape(condition)
            scattered_shape = tf.concat(
                [batch_shape, tf.shape(true_output)[tf.rank(batch_shape):]], 0)
            true_scatter = tf.scatter_nd(
                indices=tf.cast(true_indices, tf.int32),
                updates=true_output,
                shape=scattered_shape)
            false_scatter = tf.scatter_nd(
                indices=tf.cast(false_indices, tf.int32),
                updates=false_output,
                shape=scattered_shape)
            return true_scatter + false_scatter

        result = nest.pack_sequence_as(
            structure=true_outputs,
            flat_sequence=[
                scatter_outputs(true_single_output, false_single_output) for true_single_output, false_single_output
                in zip(nest.flatten(true_outputs), nest.flatten(false_outputs))]
        )
        return result


    def forward(self, hid_iter, n, false_idx, neg_idx):

        # select n-th column from left and right child lists (corresponds to n-th position in batch word sentences) and from
        # this column select only that elements that are at neg_idx position
        # left_right_idx shape: (k, ); k - number of elements in neg_idx
        left_idx = tf.reshape(
            tf.gather_nd(self.leftChild,
                         tf.stack([neg_idx, tf.fill(tf.shape(neg_idx), n)],
                                  axis=-1)
                         ),
            [-1]
        )
        right_idx = tf.reshape(
            tf.gather_nd(self.rightChild,
                         tf.stack([neg_idx, tf.fill(tf.shape(neg_idx), n)],
                                  axis=-1)
                         ),
            [-1]
        )

        def forw_cond(left_iter, right_iter, i):
            return tf.less(i, tf.shape(neg_idx)[0])

        def forw_body(left_iter, right_iter, i):
            left_iter = left_iter.write(i,
                                        tf.gather(hid_iter.read(left_idx[i]),
                                                  neg_idx[i])
                                        )
            right_iter = right_iter.write(i,
                                          tf.gather(hid_iter.read(right_idx[i]),
                                                    neg_idx[i])
                                          )
            i = tf.add(i, 1)
            return left_iter, right_iter, i

        # for each "element" in left_idx and right_idx select tensor from TensorArray that are at "element" position
        # and from that tensor select vector that corresponds to neg_idx index - batch sentense number
        left_output, right_output, _ = tf.while_loop(cond=forw_cond,
                                                     body=forw_body,
                                                     loop_vars=[tf.TensorArray(dtype=tf.float32,  # init values
                                                                               size=0,
                                                                               dynamic_size=True,
                                                                               clear_after_read=False,
                                                                               infer_shape=False),
                                                                tf.TensorArray(dtype=tf.float32,  # init values
                                                                               size=0,
                                                                               dynamic_size=True,
                                                                               clear_after_read=False,
                                                                               infer_shape=False),
                                                                tf.constant(0, dtype=tf.int32)])

        # shape: (n, 1, d) : n - size of neg_idx, d - embedding vector size
        left_output, right_output = left_output.stack(), right_output.stack()

        # left_output= tf.Print(left_output, ["left_idx:  ", left_idx,
        #                                     "left_idx shape:  ", tf.shape(left_idx),
        #                                     "left_output before shape: ", tf.shape(left_output),
        #                                     "right_output before shape: ", tf.shape(right_output)])

        # remove unnecessary dimension
        left_output = tf.cond(pred=tf.equal(tf.shape(left_output)[0], 1),
                              true_fn=lambda: left_output[0],
                              false_fn=lambda: tf.squeeze(left_output))   # remove outer [] brackets; [[[...], [...]]] => [[...], [...]]

        right_output = tf.cond(pred=tf.equal(tf.shape(right_output)[0], 1),
                               true_fn=lambda: right_output[0],
                               false_fn=lambda: tf.squeeze(right_output))

        # right_left_output shape: (n, d)

        # shape: (n, d)
        return tf.nn.tanh(self.nonlinear_mul(left_output, self.W_11, left_output) + \
                            self.nonlinear_mul(right_output, self.W_22, right_output) + \
                            self.nonlinear_mul(left_output, self.W_12, right_output) + \
                            self.dotVecMat(left_output, self.W_1[0]) + \
                            self.dotVecMat(right_output, self.W_2) + \
                            self.W_1[1])


    def body(self, hid_iter, n):

        def gather_cols(params, indices, name=None):
            """Gather columns of a 2D tensor.

            Args:
                params: A 2D tensor.
                indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
                name: A name for the operation (optional).

            Returns:
                A 2D Tensor. Has the same type as ``params``.
            """
            with tf.op_scope([params, indices], name, "gather_cols") as scope:
                # Check input
                params = tf.convert_to_tensor(params, name="params")
                indices = tf.convert_to_tensor(indices, name="indices")
                try:
                    params.get_shape().assert_has_rank(2)
                except ValueError:
                    raise ValueError('\'params\' must be 2D.')
                try:
                    indices.get_shape().assert_has_rank(1)
                except ValueError:
                    raise ValueError('\'params\' must be 1D.')

                # Define op
                p_shape = tf.shape(params)
                p_flat = tf.reshape(params, [-1])
                i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                               [-1, 1]) + indices, [-1])
                return tf.reshape(tf.gather(p_flat, i_flat),
                                  [p_shape[0], -1])

        # select n-th column from self.words (n-th word in each sentence in batch); shape (batch_sz, )
        w = tf.squeeze(gather_cols(self.words, [n]))
        #w = tf.Print(w, ["n: ", n, "w: ", w, tf.shape(w)])

        # get position of "-1" elements (that means Node has no word) in w
        # neg_idx shape (k, 1); k - number of neg indices; e.g. [[13][14]] - shape (2, 1)
        neg_idx = tf.cast(tf.where(tf.equal(w, -1)), tf.int32)

        # slicing_where returns not position based on condition but actual elements. but for forward function we need
        # positions (because all element are the same "-1"). first we check is it at least one "-1" element in w. if
        # no - select appropriate word embedding vectors, otherwise - use slicing_where - true_branch select appropriate
        # word embedding vectors and false_branch pass TensorArray with results, n - our current position in batch
        # sentences (we loop over sentence words - from 1-st to last) and ! indexes ! of "-1" elements
        hidden_value = tf.cond(pred=tf.equal(tf.size(neg_idx), 0),
                               true_fn=lambda: tf.nn.embedding_lookup(self.W_embed, w),
                               false_fn=lambda: self.slicing_where(condition=tf.math.greater_equal(w, 0), # w == 0 is (0, 0, ...) vector
                                                                   full_input=w,
                                                                   true_branch=lambda true_w: tf.nn.embedding_lookup(self.W_embed,
                                                                                                                     true_w),
                                                                   false_branch=lambda false_w: self.forward(hid_iter,
                                                                                                             n, false_w,
                                                                                                             neg_idx)))

        hid_iter = hid_iter.write(n, hidden_value)  # write new value to TensorArray
        n = tf.add(n, 1)  # increment step
        return hid_iter, n


    def model_initializer(self, V, K, nonlin_func, optimizer="adam", optimizer_args=(1e-5, 0.99, 0.999), reg=1e-2,
                          train_inner_nodes=False, load_w=None):

        self.V = V
        self.K = K
        self.f = nonlin_func

        self.words = tf.placeholder(tf.int32, shape=(None, None), name="words")
        self.leftChild = tf.placeholder(tf.int32, shape=(None, None), name="leftChd")
        self.rightChild = tf.placeholder(tf.int32, shape=(None, None), name="rightChd")
        self.labels = tf.placeholder(tf.int32, shape=(None, None), name="labels")
        self.batch_sz = tf.placeholder(tf.int32,  name="batch_sz")

        if load_w:
            l = lambda arr_num: tf.Variable(load_w[arr_num])
            self.W_embed = l("arr_0")
            self.W_11, self.W_12, self.W_22  = l("arr_1"), l("arr_2"), l("arr_3")
            self.W_1 = [l("arr_4"), l("arr_5")]
            self.W_2 = l("arr_6")
            self.W_out = [l("arr_7"), l("arr_8")]

        else:
            h = lambda size_tuple: self.helper(size_tuple)
            self.W_embed = h((self.V, self.D))[0]  # only weights, no bias
            # add first row of zeros - keras.pad_sequences "0" element
            self.W_embed = tf.concat([tf.constant([[0] * self.D], dtype=tf.float32), self.W_embed], axis=0)
            self.W_11, self.W_12, self.W_22 = h((self.D, self.D, self.D))[0], \
                                              h((self.D, self.D, self.D))[0], \
                                              h((self.D, self.D, self.D))[0],  # only weights, no bias
            self.W_1 = h((self.D, self.D))
            self.W_2 = h((self.D, self.D))[0]  # only weights, no bias
            self.W_out = h((self.D, self.K))

        self.params = [self.W_embed, self.W_11, self.W_12, self.W_22, self.W_1, self.W_2, self.W_out]

        def condition(hid_iter, n):
            return tf.less(n, tf.shape(self.words)[1])

        hid_output, _ = tf.while_loop(
            cond=condition,  # when to stop
            body=self.body,  # what to do
            loop_vars=[tf.TensorArray(dtype=tf.float32,  # init values
                                      size=0,
                                      dynamic_size=True,
                                      clear_after_read=False,
                                      infer_shape=False),
                       tf.constant(0, dtype=tf.int32)],
            parallel_iterations=1
        )

        # hid_output tensor after stack size:  [num_of_words x batch_sz x D_embed_sz];
        # num_of_words - length of sentence
        # batch_sz - number of sentences in batch
        # D_embed_sz - shape of word embedding vector
        hid_output = hid_output.stack()

        # logits size: [num_of_words x batch_sz x K_output_classes]
        logits = tf.tensordot(hid_output, self.W_out[0], axes=[[2], [0]]) + self.W_out[1]
        # self.prediction size: [num_of_words x batch_sz]
        self.prediction = tf.argmax(logits, axis=2)

        ''' sum only weights '''
        rcost = reg * sum([tf.nn.l2_loss(w) for w in [self.W_embed, self.W_11, self.W_12, self.W_22, self.W_2]])
        ''' sum weights and biases '''
        rcost += reg * sum([tf.nn.l2_loss(w_b) for w_and_b in [self.W_1, self.W_out] for w_b in w_and_b])

        if train_inner_nodes:
            ''' (m,k), where m - number of such occurences and k=2, k: (num_of_sent_in_batch, position_of_word_in_sentence)'''
            labels = tf.transpose(self.labels, [1, 0])
            idx = tf.where(labels > 0)
            logits_ = tf.gather_nd(logits, idx)
            labels_ = tf.gather_nd(labels, idx)
        else:
            ''' (k, ) shape tensor - k=batch_sz - num of sentences in batch '''
            logits_ = logits[:, -1]
            labels_ = self.labels[:, -1]

        #labels_ = tf.Print(labels_, ["logits_: ", logits_,
        #                             "logits_ shape: ", tf.shape(logits_),
        #                             "labels_: ", labels_,
        #                             "labels_ shape: ", tf.shape(labels_),])

        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits_,
                labels=labels_)
        ) + rcost

        self.train_op = self.optimizer(optimizer, optimizer_args).minimize(self.cost)


    def make_lists(self, dataset, batch_num, batch_sz):
        M = dataset[batch_num * batch_sz: (batch_num + 1) * batch_sz]
        words, leftCh, rightCh, labels = list(zip(*[M[i] for i in range(len(M))]))

        sequence_length = max(len(x) for x in words)

        words, leftCh, rightCh, labels = list(
                       map(lambda x: tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=sequence_length),
                           [words, leftCh, rightCh, labels])
        )

        labels = np.array(labels)

        # leftCh and rightCh lists contain idx of tree Nodes and this numbers are indexes of TensorArray tensors
        # after padding short sentences, to keep this ability we have to add j = amount of zeros at the beginning
        # and subtract 1, because buildParseTree start Node idx from 1 (to differentiate padding zero and "true" zero
        # Node idx )  [0, 0, -1, -1, 1, 3] => (num of zeros = 2) => [0, 0, -1, -1, 1+2-1, 3+2-1]
        leftCh = list(map(list,
                          [np.add(np.array(i), np.count_nonzero(np.array(i) == 0) - 1, where=(np.array(i) > 0))
                           for i in leftCh]))
        rightCh = list(map(list,
                           [np.add(np.array(i), np.count_nonzero(np.array(i) == 0) - 1, where=(np.array(i) > 0))
                            for i in rightCh]))
        return words, leftCh, rightCh, labels


    def batch_fit(self, tree_lsts, V, K, nonlin_func, optimizer, optimizer_args, epochs, reg, train_inner_nodes, batch_sz):

        #self.batch_sz = batch_sz

        # got list of sentence arrays with words, leftCh, rightCh, label lists
        train_set, valid_set = map(list, sklearn.model_selection.train_test_split(
                                                                                  np.array(tree_lsts),
                                                                                  test_size=0.15,
                                                                                  random_state=42)
                                   )

        print("TRAIN SET LENGTH: ", len(train_set))

        # sort train sentences by its length to get minimum number of "zeros" in batch  after keras.padding
        train_set.sort(key=lambda x: len(x[0]))
        valid_set.sort(key=lambda x: len(x[0]))

        self.model_initializer(V, K, nonlin_func, optimizer, optimizer_args, reg, train_inner_nodes)
        self.session.run(tf.global_variables_initializer())

        n_train_batch = (len(train_set) + batch_sz - 1) // batch_sz
        n_valid_batch = (len(valid_set) + batch_sz - 1) // batch_sz

        costs = [0] * epochs
        train_accuracy = [0] * epochs
        valid_accuracy = [0] * epochs

        for epoch in range(epochs):
            t0 = datetime.datetime.now()

            train_correct, valid_correct = [], []

            print("TRAINING")
            for batch_num in range(n_train_batch):
                words, leftCh, rightCh, labels = self.make_lists(train_set, batch_num, batch_sz)

                predict, c, _ = self.session.run([self.prediction, self.cost, self.train_op],
                                                 feed_dict={self.words: words,
                                                            self.leftChild: leftCh,
                                                            self.rightChild: rightCh,
                                                            self.labels: labels,
                                                            self.batch_sz: len(words)})

                #print("c: ", c, "predict: ",  predict)
                costs[epoch] += c
                train_correct.extend(predict[-1] == labels[:, -1])

            train_accuracy[epoch] = round(sum(train_correct) / float(len(train_correct)), 3)


            print("VALIDATION")
            for batch_num in range(n_valid_batch):
                words, leftCh, rightCh, labels = self.make_lists(valid_set, batch_num, batch_sz)

                predict = self.session.run(self.prediction, feed_dict={self.words: words,
                                                                       self.leftChild: leftCh,
                                                                       self.rightChild: rightCh,
                                                                       self.labels: labels,
                                                                       self.batch_sz: len(words)})


                valid_correct.extend(predict[-1] == labels[:, -1])

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
        # print("W_11: ", actual_params[6][0])

        np.savez(filename, *saved_lst)


    @staticmethod
    def load_weights(filename):
        npz = np.load(filename)
        ''' W_embed, W_11, W_12, W_22, W_1, W_2, W_out, D, f_nonlin, V, K '''
        D, f_nonlin, V, K = int(npz["arr_9"]), str(npz["arr_10"]), int(npz["arr_11"]), int(npz["arr_12"])
        tnn = model = TNN(D)
        tnn.model_initializer(V, K, f_nonlin, load_w=npz)

        return tnn


    def predict(self, tree_set, batch_sz, k=False):
        ''' return root node label prediction '''

        n_pred_batch = (len(tree_set) + batch_sz - 1) // batch_sz

        #words, leftCh, rightCh, labels = tree
        if k:
            self.session.run(tf.global_variables_initializer())

        prediction_lst = []

        for batch_num in range(n_pred_batch):
            words, leftCh, rightCh, labels = self.make_lists(tree_set, batch_num, batch_sz)

            prediction_lst.extend(self.session.run(self.prediction, feed_dict={self.words: words,
                                                                               self.leftChild: leftCh,
                                                                               self.rightChild: rightCh,
                                                                               self.labels: labels,
                                                                               self.batch_sz: len(words)})[-1]
                                 )
        return prediction_lst


    def score_prediction(self, p, l):
        """ return accuracy and f_1 score """
        return np.sum((np.equal(p, l))) / (1.0 * len(l)), f1_score(p, l, average=None).mean()


if __name__ == "__main__":
    train_trees, test_trees, word2idx = get_data("recursive_train.txt", "recursive_test.txt")

    index2word = {index: word for word, index in word2idx.items()}

    f_lst = lambda tree: [get_tree_lst(t, -1) for t in tree]

    train_lst = f_lst(train_trees)
    test_lst = f_lst(test_trees)

    V = len(word2idx)  # +1 - add "0" word and 0 pos into W_embed - to account absence of word in sentence after keras pad_sequences
    K = 5 + 1          # +1 - add "0" label and 0 pos into W_embed - to account absence of word in sentence after keras pad_sequences

    # words: [97, 7, 98, -1, 3, 99, 100, -1, -1, 101, -1, 34, -1, -1, -1]
    # leftCh: [-1, -1, -1, 5, -1, -1, -1, 12, 10, -1, 9, -1, 8, 4, 2]
    # rightCh: [-1, -1, -1, 6, -1, -1, -1, 13, 11, -1, 14, -1, 15, 7, 3]
    # labels: ['2', '2', '2', '2', '2', '2', '4', '3', '3', '2', '4', '2', '3', '3', '3']

    model = TNN(10)
    session = tf.InteractiveSession()
    model.set_session(session)

    model.batch_fit(train_lst[:1500], V, K, nonlin_func="tanh", optimizer="adagrad", optimizer_args=(8e-3,), epochs=30,
                    reg=1e-3, train_inner_nodes=True, batch_sz=15)


    model.save_weights("recursiveNN_weights_batch.npz")

    print(model.predict(test_lst[:35], batch_sz=15))

    print("test accuracy: %f test f1 score: %f", (model.score_prediction(model.predict(test_lst[:500], batch_sz=15), list(map(operator.itemgetter(-1), np.array(test_lst)[:500, 3])))))


    model_1 = TNN.load_weights("recursiveNN_weights_batch.npz")
    session = tf.InteractiveSession()
    model_1.set_session(session)

    print(model_1.predict(test_lst[:35], batch_sz=15, k=True))

    print("test accuracy: %f test f1 score: %f", (model_1.score_prediction(model_1.predict(test_lst[:500], batch_sz=15, k=True), list(map(operator.itemgetter(-1), np.array(test_lst)[:500, 3])))))


