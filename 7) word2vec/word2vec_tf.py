import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.spatial.distance import cdist as cdist
from sklearn.metrics.pairwise import pairwise_distances as pairwise_distances
import string
import itertools
import tensorflow as tf
from glob import glob
from datetime import datetime



fun = lambda f_name: os.path.realpath(os.path.join(os.getcwd(), f_name)).replace("\\", "/")


def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))


def neg_distribution(sentences, num_of_words):
    word_freq = np.bincount(np.fromiter(itertools.chain(*sentences), int))

    ''' smoothing '''
    p_neg = (word_freq ** 0.75 / (word_freq ** 0.75).sum())
    return p_neg


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
                                #if word not in all_word_counts:
                                #    all_word_counts[word] = 0
                                #all_word_counts[word] += 1
                                all_word_counts[word] = all_word_counts.get(word, 0) + 1
        except UnicodeDecodeError:
            continue
    print("finished counting")
    print("dict size: ", len(all_word_counts))

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



class Word2Vec:
    def __init__(self, context_sz, epochs, embed_sz):
        self.context_sz = context_sz
        self.epochs = epochs
        self.D = embed_sz

    def get_context(self, sentence, pos):
        start = max(0, pos-self.context_sz)
        last = min(sentence.shape[0], pos+self.context_sz)
        return np.concatenate([sentence[start : pos], sentence[pos+1 : last]])

    def forward(self, input, target, label):
        # (1, D)  (D, N)
        Z = tf.nn.embedding_lookup(self.W1, input)
        prediction = tf.matmul(Z, tf.transpose(tf.nn.embedding_lookup(self.W2, target)))

        if label == 1:
            return tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones(tf.shape(prediction)), logits=tf.cast(prediction, tf.float32))
        elif label == 0:
            return tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros(tf.shape(prediction)), logits=tf.cast(prediction, tf.float32))

    def optimizer(self, optimizer, opt_args):
        if optimizer.lower() == "adam":
            optimizer = tf.train.AdamOptimizer
        elif optimizer.lower() == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer
        elif optimizer.lower() == "momentum":
            optimizer = tf.train.MomentumOptimizer
        elif optimizer.lower() == "proximal":
            optimizer = tf.train.ProximalGradientDescentOptimizer
        else:
            raise ValueError('UNSUPPORTED OPTIMIZER TYPE')

        try:
            return optimizer(*opt_args)
        except ValueError:
            print("Uncorrect arguments for " + optimizer + " optimizer")

    def set_session(self, session):
        self.session = session

    def train(self, vocab_size, optimizer, opt_args, wiki_files):
        sentences, word2idx = get_wiki(vocab_size, wiki_files=wiki_files)

        num_of_words = len(word2idx)

        costs = []

        init_weights = lambda z1, z2: np.random.randn(z1, z2)
        '''  .T - thats because tf.nn.embedding_lookup in forward() method select row but not column'''
        self.W1, self.W2 = tf.Variable(init_weights(num_of_words, self.D)), tf.Variable(init_weights(self.D, num_of_words).T)
        middle_word_tf = tf.placeholder(tf.int32, shape=(None, ))
        context_tf = tf.placeholder(tf.int32, shape=(None, ))
        neg_word_tf = tf.placeholder(tf.int32, shape=(None, ))

        loss = tf.reduce_mean(self.forward(middle_word_tf, context_tf, 1)) + tf.reduce_mean(self.forward(middle_word_tf, neg_word_tf, 0))

        train_optimizer = self.optimizer(optimizer, opt_args).minimize(loss)

        self.session.run(tf.global_variables_initializer())

        ''' distribution to get negative samples '''
        neg_probability = neg_distribution(sentences, num_of_words)

        threshold = 1e-5
        drop_probability = 1 - np.sqrt(threshold / neg_probability + 1e-10)

        for e in range(self.epochs):
            print("start epoch: ", e)
            t_start = datetime.now()
            np.random.shuffle(sentences)
            cost = 0
            middle_word, context, neg_word = np.array([]), np.array([]), np.array([])

            for num, sentence in enumerate(sentences):
                s = np.array(sentence)[np.random.random() > np.take(drop_probability, tuple(sentence))]
                if s.shape[0] >= 2:

                    ''' randomly choose order of selected words '''
                    for position in np.random.choice(s.shape[0], s.shape[0], replace=False):
                        middle_word = np.append(middle_word, s[position])
                        context = np.concatenate((context, self.get_context(s, position)))
                        neg_word = np.append(neg_word, np.random.choice(num_of_words, p=neg_probability, replace=False))


                    if middle_word.shape[0] > 128:
                        _, c = self.session.run((train_optimizer, loss), feed_dict={middle_word_tf: middle_word,
                                                                                    context_tf: context,
                                                                                    neg_word_tf: neg_word})
                        cost += c
                        middle_word, context, neg_word = np.array([]), np.array([]), np.array([])


            print("epoch: ", e, " cost: ", cost, " elapsed time: ", datetime.now() - t_start)
            costs.append(cost)

        plt.plot(costs)
        plt.xlabel("epoch")
        plt.xlabel("cost")
        plt.grid()
        plt.show()

        with open('word2idx_tf.json', 'w') as f:
            json.dump(word2idx, f)

        W1, W2 = self.session.run((self.W1, self.W2))
        np.savez('weights_tf.npz', W1, W2.T)

        return word2idx, self.session.run((self.W1, self.W2))

    @staticmethod
    def load_model():
        with open('word2idx_tf.json') as f:
            word2idx = json.load(f)
        npz = np.load('weights_tf.npz')
        W1, W2 = npz['arr_0'], npz['arr_1']
        return word2idx, W1, W2




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
    #print("HAY: ", sort_cos[:10], word2idx[find_word])
    print("position of a true word: ", true_position(sort_cos, word2idx[find_word]))
    print("euclid: ")
    print("euclid %s - %s = %s - %s" % (word1, word2, idx2word[sort_euclid[0]], word3))
    print("closest words:")
    print([idx2word[i] for i in sort_euclid[:5]])
    print("position of a true word: ", true_position(sort_euclid, word2idx[find_word]))


if __name__ == "__main__":
    c = Word2Vec(context_sz=5, epochs=24, embed_sz=50)
    c.set_session(tf.Session())
    word2idx, W_1, W_2 = c.train(vocab_size=20000, optimizer="momentum", opt_args=(1e-1, 0.9,),
                                 wiki_files=fun('enwiki/*/wiki_*'))

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
        find_analogy(*word_pair, word2idx, (W_1 + W_2.T) / 2)

    word2idx, W_1, W_2 = Word2Vec.load_model()
    W_embed = (W_1 + W_2.T) / 2

    for word_pair in word_lst:
        find_analogy(*word_pair, word2idx, (W_1 + W_2.T) / 2)
