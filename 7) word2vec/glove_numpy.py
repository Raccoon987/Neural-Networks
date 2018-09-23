import os
import json
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.utils import shuffle
from glob import glob
import scipy
from scipy.sparse import lil_matrix
import string
from scipy.spatial.distance import cdist as cdist

def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))

def get_wiki(vocab_size, wiki_files):
    files = glob(wiki_files)
    all_word_counts = {}
    for f in files:
        try:
            for line in open(f):
                if line and line[0] not in '<[*-|=\{\}':
                    for sentence in line.split(". "):
                        s = remove_punctuation(sentence).lower().split()
                        if len(s) > 1:
                            for word in s:
                                all_word_counts[word] = all_word_counts.get(word, 0) + 1
        except UnicodeDecodeError:
            continue
    print("finished counting")

    vocab_size = min(vocab_size, len(all_word_counts))
    all_word_counts = sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True)

    top_words = [w for w, count in all_word_counts[:vocab_size-1]] + ['<UNK>']
    word2idx = {w:i for i, w in enumerate(top_words)}

    sents = []
    for f in files:
        try:
            for line in open(f):
                if line and line[0] not in '<[*-|=\{\}':
                    for sentence in line.split(". "):
                        s = remove_punctuation(line).lower().split()
                        if len(s) > 1:
                            sents.append([word2idx.get(word, word2idx['<UNK>']) for word in s])

        except UnicodeDecodeError:
            continue
    return sents, word2idx


class Glove:
    def __init__(self, embed_sz, context_sz):
        self.context_sz = context_sz
        self.D = embed_sz

    def build_cooccur_mtx(self, sentences, cooccur_mtx_path):

        def save_sparse_matrix(filename, x):
            x_coo = x.tocoo()  # return a COOrdinate representation of this matrix
            np.savez(filename, row=x_coo.row, col=x_coo.col, data=x_coo.data, shape=x_coo.shape)

        def load_sparse_matrix(filename):
            y = np.load(filename)
            z = scipy.sparse.coo_matrix((y['data'], (y['row'], y['col'])), shape=y['shape'])
            return z.tolil()

        if not os.path.exists(cooccur_mtx_path):
            print("matrix shape: ", self.V, " x ", self.V)
            X_cooccur = lil_matrix(np.zeros((self.V, self.V)))
            for sent_idx, sentence in enumerate(sentences):
                if sent_idx % 10000 == 0:
                    print("Building co-occurrence matrix: on line * 10000 %i: ", sent_idx // 10000)

                for word_idx, center_word in enumerate(sentence):
                    left_context_words = sentence[max(0, word_idx - self.context_sz) : word_idx]
                    left_context_len = len(left_context_words)

                    right_context_words = sentence[word_idx + 1: min(word_idx + self.context_sz, len(sentence))]

                    for left_idx, left_cont_word in enumerate(left_context_words):
                        distance = left_context_len - left_idx        # distance from center word
                        increment = 1.0 / float(distance)             # weighted increment
                        X_cooccur[center_word, left_cont_word] += increment

                    for right_idx, right_cont_word in enumerate(right_context_words):
                        distance = right_idx + 1                      # distance from center word
                        increment = 1.0 / float(distance)             # weighted increment
                        X_cooccur[center_word, right_cont_word] += increment

            save_sparse_matrix(cooccur_mtx_path, X_cooccur)

        else:
            #X_cooccur = np.load(cooccur_mtx_path)
            X_cooccur = load_sparse_matrix(cooccur_mtx_path)
            print("loaded matrix",  X_cooccur.shape, type(X_cooccur))

        return X_cooccur


    def train(self, vocab_size, wiki_files, cooccur_mtx_path, lr=1e-4, reg=0.1, xmax=100, alpha=0.75, epochs=10, fit_method = "grad_descent"):
        sentences, word2idx = get_wiki(vocab_size, wiki_files=wiki_files)

        self.V = len(word2idx)  # number of unique words

        cooccur_mtx = (self.build_cooccur_mtx(sentences, cooccur_mtx_path))

        if not isinstance(cooccur_mtx, np.ndarray):
            cooccur_mtx = cooccur_mtx.toarray()

        print(type(cooccur_mtx), cooccur_mtx.shape)
        f_X = np.where(np.greater(cooccur_mtx, xmax),
                       np.ones((self.V, self.V)),
                       (cooccur_mtx / xmax) ** alpha)
        #f_X = (cooccur_mtx / xmax) ** alpha if cooccur_mtx < xmax else 1
        log_X = np.log(cooccur_mtx + 1)

        # init matricies
        init_weights = lambda z1, z2: np.random.randn(z1, z2) / np.sqrt(z1 + z2)
        W, U = init_weights(self.V, self.D), init_weights(self.V, self.D)
        b, c = np.zeros(self.V), np.zeros(self.V)
        mu = log_X.mean()

        costs = []


        for e in range(epochs):
            #  J' = w_i^T.dot(w_j) + b_i + b_j - log(X_ij)
            cost_inner = (W.dot(U.T) + b.reshape(self.V, 1) + c.reshape(1, self.V) + mu) - log_X
            # cost J = f(X_{ij})*(J')^2
            costs.append((f_X * cost_inner * cost_inner).sum())
            print(costs[-1])

            if fit_method == "grad_descent":
                w_update = (f_X * cost_inner).dot(U)
                W -= lr * w_update * (1 + reg)
                #print("f_X shape: ", f_X.shape, "cost_inner shape: ", cost_inner.shape, "b shape: ", b.shape)
                b -= lr * np.sum(f_X * cost_inner, axis=1)

                u_update = (f_X * cost_inner).dot(W)
                U -= lr * u_update * (1 + reg)
                c -= lr * np.sum(f_X * cost_inner, axis=0)

            elif fit_method == "ALS":
                for i in range(self.V):
                    ''' w(i) = (sum(j)[f(X(i, j))*u(j)*u(j).T] + lambda*I)^-1 * (sum(j)[f(X(i, j))*[log(X(i, j) - b(i) - c(j) - mu)].dot(u(j))) '''
                    A = reg * np.eye(self.D) + (f_X[i, :] * U.T).dot(U)
                    a = (f_X[i, :] * (log_X[i, :] - b[i] - c - mu)).dot(U)
                    W[i] = np.linalg.solve(A, a)

                for i in range(self.V):
                    denom = (f_X[i, :].sum()) * (1 + reg)
                    num = f_X[i, :].dot(log_X[i, :] - W[i].dot(U.T) - c - mu)
                    b[i] = num / denom

                for j in range(self.V):
                    A = reg * np.eye(self.D) + (f_X[:, j] * W.T).dot(W)
                    a = (f_X[:, j] * (log_X[:, j] - b - c[j] - mu)).dot(W)
                    U[i] = np.linalg.solve(A, a)

                for j in range(self.V):
                    denom = (f_X[:, j].sum()) * (1 + reg)
                    num = f_X[:, j].dot(log_X[:, j] - - W[i].dot(U.T) - b - mu)
                    c[j] = num / denom

            else:
                raise NameError('methods should be: grad_descent or ALS')


        self.W, self.U = W, U

        plt.plot(costs)
        plt.xlabel("epoch")
        plt.ylabel("cost")
        plt.grid()
        plt.show()

        with open('glove_np_word2idx.json', 'w') as f:
            json.dump(word2idx, f)
        np.savez('glove_np_weights.npz', self.W, self.U)

        return word2idx, self.W, self.U

    @staticmethod
    def load_model():
        with open('glove_np_word2idx.json') as f:
            word2idx = json.load(f)
        npz = np.load('glove_np_weights.npz')
        W, U = npz['arr_0'], npz['arr_1']
        return word2idx, W, U




def find_analogy(word1, word2, find_word, word3, word2idx, W_embed):
    if (word1 not in word2idx) or (word2 not in word2idx) or (word3 not in word2idx) or (find_word not in word2idx):
        print("unfortunately one of this words is not in text corpus")
        return

    print("actual equation must be: ", "%s - %s = %s - %s" % (word1, word2, find_word, word3))
    vec_dim, num_of_words = W_embed.shape[1], W_embed.shape[0]

    f_idx = lambda word: word2idx[word]
    f_vec = lambda word: W_embed[word2idx[word]]
    idx_1, idx_2, idx_3 = map(f_idx, [word1, word2, word3])
    vec_1, vec_2, vec_3 = map(f_vec, [word1, word2, word3])

    vector = vec_1 - vec_2 + vec_3

    dist_cosine = cdist(vector.reshape(1, vec_dim), W_embed, metric='cosine').reshape(num_of_words)
    dist_euclid = cdist(vector.reshape(1, vec_dim), W_embed, metric='euclidean').reshape(num_of_words)

    ''' set distances that corresponds to words: word1, word2, word3 to np.inf '''
    dist_cosine[[idx_1, idx_2, idx_3]] = np.inf
    dist_euclid[[idx_1, idx_2, idx_3]] = np.inf

    sort_cos = np.argsort(dist_cosine)
    sort_euclid = np.argsort(dist_euclid)

    ''' find position of true answer '''
    true_position = lambda sort_arr, index: np.where(sort_arr == index)[0][0]

    idx2word = {i: w for w, i in word2idx.items()}
    print("cosine: ")
    print("cosine %s - %s = %s - %s" % (word1, word2, idx2word[sort_cos[0]], word3))
    print("closest words:")
    print([idx2word[i] for i in sort_cos[:5]])
    print("position of a true word: ", true_position(sort_cos, word2idx[find_word]))
    print("euclid: ")
    print("euclid %s - %s = %s - %s" % (word1, word2, idx2word[sort_euclid[0]], word3))
    print("closest words:")
    print([idx2word[i] for i in sort_euclid[:5]])
    print("position of a true word: ", true_position(sort_euclid, word2idx[find_word]))



if __name__ == "__main__":
    model = Glove(embed_sz=100, context_sz=5)
    word2idx, W, U = model.train(vocab_size=2000,
                               wiki_files='C:/Users/Perederiy/PycharmProjects/Deep_NLP/Deep_NLP/enwiki/*/wiki_*',
                               cooccur_mtx_path="C:/Users/Perederiy/PycharmProjects/Deep_NLP/Deep_NLP/coocur_mtx_wiki.npz",
                               lr=1e-4,
                               reg=0.1,
                               xmax=100,
                               alpha=0.75,
                               epochs=25,
                               fit_method="ALS")

    word_lst = [['king', 'man', 'queen', 'woman'], ['king', 'prince', 'queen', 'princess'],
                ['miami', 'florida', 'dallas', 'texas'], ['einstein', 'scientist', 'picasso', 'painter'],
                ['japan', 'sushi', 'germany', 'bratwurst'], ['man', 'woman', 'he', 'she'],
                ['man', 'woman', 'uncle', 'aunt'], ['man', 'woman', 'brother', 'sister'],
                ['man', 'woman', 'husband', 'wife'], ['man', 'woman', 'actor', 'actress'],
                ['man', 'woman', 'father', 'mother'], ['heir', 'heiress', 'prince', 'princess'],
                ['nephew', 'niece', 'uncle', 'aunt'], ['france', 'paris', 'japan', 'tokyo'],
                ['france', 'paris', 'china', 'beijing'], ['february', 'january', 'december', 'november'],
                ['france', 'paris', 'germany', 'berlin'], ['week', 'day', 'year', 'month'],
                ['week', 'day', 'hour', 'minute'], ['france', 'paris', 'italy', 'rome'],
                ['paris', 'france', 'rome', 'italy'], ['france', 'french', 'england', 'english'],
                ['japan', 'japanese', 'china', 'chinese'], ['china', 'chinese', 'america', 'american'],
                ['japan', 'japanese', 'italy', 'italian'], ['japan', 'japanese', 'australia', 'australian'],
                ['walk', 'walking', 'swim', 'swimming']]

    for word_pair in word_lst:
        find_analogy(*word_pair, word2idx, (W + U) / 2)

    word2idx, W_1, W_2 = Glove.load_model()
    W_embed = (W_1 + W_2) / 2

    for word_pair in word_lst:
        find_analogy(*word_pair, word2idx, W_embed)