import numpy as np
from scipy.special import expit as sigmoid
import matplotlib.pyplot as plt
import json
from scipy.spatial.distance import cdist as cdist
from sklearn.metrics.pairwise import pairwise_distances as pairwise_distances
import nltk
import string
import operator
import itertools
from glob import glob

KEEP_WORDS = set(['king', 'man', 'queen', 'woman', 'italy', 'rome', 'france', 'paris',
                  'london', 'britain', 'england'])


def get_wiki(vocab_size, wiki_files):
    files = glob(wiki_files)
    all_word_counts = {}
    for f in files:
        print(f, end=" ")
        try:
            for line in open(f):
                if line and line[0] not in '<[*-|=\{\}':
                    for sentence in line.split(". "):
                        s = remove_punctuation(sentence).lower().split()
                        if len(s) > 1:
                            for word in s:
                                if word not in all_word_counts:
                                    all_word_counts[word] = 0
                                all_word_counts[word] += 1
        except UnicodeDecodeError:
            continue
    print("finished counting")

    V = min(vocab_size, len(all_word_counts))
    all_word_counts = sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True)

    top_words = [w for w, count in all_word_counts[:V-1]] + ['<UNK>']
    word2idx = {w:i for i, w in enumerate(top_words)}
    unk = word2idx['<UNK>']

    sents = []
    for f in files:
        try:
            for line in open(f):
                if line and line[0] not in '<[*-|=\{\}':
                    for sentence in line.split(". "):
                        s = remove_punctuation(line).lower().split()
                        if len(s) > 1:
                            sent = [word2idx[w] if w in word2idx else unk for w in s]
                            sents.append(sent)
        except UnicodeDecodeError:
            continue
    return sents, word2idx

def get_sentence():
    ''' return list of lists containing sentences; each sentence - tokenized words and punktuation like ["i", "am", "," ...] '''
    return nltk.corpus.brown.sents()


def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))


def index_sentence():
    word2idx, wordidx_count = {}, {}
    sentences_idx = []
    for sentence in get_sentence():
        sentences_idx.append([])
        for word in [word for word in [remove_punctuation(s) for s in sentence] if word]: #remove punktuation and " " symbols::
            word = word.lower()
            if word not in word2idx:
                word2idx[word] = len(word2idx)
            wordidx_count[ word2idx[word] ] = wordidx_count.get( word2idx[word] , 0) + 1
            sentences_idx[-1].append(word2idx[word])
    print("Vocab size:", len(word2idx))
    ''' return list of sentences coded by unique numbers; dict word=>index; dict of word frequencies (words encoded with numbers) '''
    return sentences_idx, word2idx, wordidx_count


''' get indexed sentences with dictionary of words limited by N most frequently used words '''
def index_sentence_limit(vocab_size=2000, keep_words=KEEP_WORDS):
    sentences_idx, word2idx, wordidx_count = index_sentence()
    ind2word = {v: k for k, v in word2idx.items()}

    for word in keep_words:
        wordidx_count[ word2idx[word] ] = float('inf')
    sorted_wi_count = sorted(wordidx_count.items(), key=operator.itemgetter(1), reverse=True)

    word2idxSmall = {}
    old_new_idx = {}
    for idx, counter in sorted_wi_count[:vocab_size]:
        word2idxSmall[ ind2word[idx] ] = len(word2idxSmall)
        old_new_idx[idx] = word2idxSmall[ ind2word[idx] ]
    ''' let 'unknown' be the last token '''
    word2idxSmall['UNKNOWN'] = len(word2idxSmall)

    ''' map old idx to new idx '''
    small_sent_idx = []
    for sentence in sentences_idx:
        if len(sentence) > 1:
            small_sent_idx.append([old_new_idx.get(item, word2idxSmall['UNKNOWN'])  for item in sentence])

    return small_sent_idx, word2idxSmall



def neg_distribution(sentences, num_of_words):
    word_freq = np.bincount(np.fromiter(itertools.chain(*sentences), int))

    ''' smoothing '''
    p_neg = (word_freq ** 0.75 / (word_freq ** 0.75).sum())
    return p_neg


class Word2Vec:
    def __init__(self, context_sz, num_neg_samples, embed_sz):
        self.context_sz = context_sz
        self.neg_sampl = num_neg_samples
        self.D = embed_sz

    def get_context(self, sentence, pos):
        start = max(0, pos-self.context_sz)
        last = min(sentence.shape[0], pos+self.context_sz)
        return np.concatenate([sentence[start : pos], sentence[pos+1 : last]])

    def grad_descent(self, input, target, label):
        prediction = sigmoid(self.W1[input].dot(self.W2[:, target]))
        ''' calculate gradients and update weights '''
        dW2 = (np.outer(self.W1[input], prediction - label)) # D x N
        dW1 = (np.sum((prediction - label) * self.W2[:, target], axis=1)) # N

        self.W2[:, target] -= self.lr * dW2
        self.W1[input] -= self.lr * dW1

        cost = label * np.log(prediction + 1e-10) + (1 - label) * np.log(1 - prediction + 1e-10)
        return cost.sum()

    def train(self, vocab_size, lr, epochs, lr_decrease, wiki_files_path):
        #sentences, word2idx = index_sentence_limit(vocab_size=vocab_size)
        sentences, word2idx = get_wiki(vocab_size, wiki_files=wiki_files_path)

        num_of_words = len(word2idx)

        costs = []

        self.lr = lr
        self.epochs = epochs

        init_weights = lambda z1, z2: np.random.randn(z1, z2)
        self.W1, self.W2 = init_weights(num_of_words, self.D), init_weights(self.D, num_of_words)

        ''' distribution to get negative samples '''
        neg_probability = neg_distribution(sentences, num_of_words)

        threshold = 1e-5
        drop_probability = 1 - np.sqrt(threshold / neg_probability + 1e-10)

        for e in range(self.epochs):
            cost = 0
            np.random.shuffle(sentences)
            for sentence in sentences:
                s = np.array(sentence)[np.random.random() > np.take(drop_probability, tuple(sentence))]
                if s.shape[0] >= 2:

                    ''' randomly choose order of selected words '''
                    for position in np.random.choice(s.shape[0], s.shape[0], replace=False):
                        middle_word = s[position]

                        context = self.get_context(s, position)
                        neg_word = np.random.choice(num_of_words, p=neg_probability, replace=False)
                        cost += self.grad_descent(middle_word, context, label=1)
                        cost += self.grad_descent(neg_word, context, label=0)

            print("epoch: ", e, " cost: ", cost)
            costs.append(cost)
            self.lr /= lr_decrease

        plt.plot(costs)
        plt.xlabel("epoch")
        plt.xlabel("cost")
        plt.grid()
        plt.show()

        with open('word2idx.json', 'w') as f:
            json.dump(word2idx, f)
        np.savez('weights.npz', self.W1, self.W2)

        return word2idx, self.W1, self.W2



    @staticmethod
    def load_model():
        with open('word2idx.json') as f:
            word2idx = json.load(f)
        npz = np.load('weights.npz')
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
    print("position of a true word: ", true_position(sort_cos, word2idx[find_word]))
    print("euclid: ")
    print("euclid %s - %s = %s - %s" % (word1, word2, idx2word[sort_euclid[0]], word3))
    print("closest words:")
    print([idx2word[i] for i in sort_euclid[:5]])
    print("position of a true word: ", true_position(sort_euclid, word2idx[find_word]))


if __name__ == "__main__":
    c = Word2Vec(context_sz=5, num_neg_samples=5, embed_sz=50)
    word2idx, W_1, W_2 = c.train(vocab_size=25000,
                                 lr=0.02,
                                 epochs= 25,
                                 lr_decrease = 1.1,
                                 wiki_files_path='C:/Users/Spectr/PycharmProjects/autoencoder/Deep_NLP/enwiki/*/wiki_*')

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


