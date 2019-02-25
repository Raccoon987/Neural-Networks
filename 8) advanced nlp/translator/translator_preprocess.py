import os, sys
from keras.preprocessing.text import Tokenizer
import numpy as np


NUM_SAMPLES = 10000  # Number of samples to train on.


def preprocess(text_file, config, embedd_file_1=None, embedd_file_2=None):
    '''
    :param file_name: file with text translation (e.g. english=>russian); tab separated sentences
    :param configs: configuration dictionary
    :param embedd_files: txt file with pretrained word vectors (optional)

    :return: tokenized origin (input), translation_input (target_input) and translation_target (target) tokenized
             matricies N x T -> N - num of sentences, T - sentence length;
             embedding matrix or matricies - if embedd_file_1, embedd_file_2 not None ;
             word to index mapping;
             config file
    '''
    input_texts = []  # sentence in original language
    target_texts = []  # sentence in target language
    target_texts_inputs = []  # sentence in target language offset by 1

    t = 0
    for line in open('en_rus_translator.txt', 'r', encoding='utf8'):
        # only keep a limited number of samples
        t += 1
        if t > NUM_SAMPLES:
            break

        # input and target are separated by tab
        if '\t' not in line:
            continue

        input_text, translation = line.rstrip().split('\t')

        target_text = translation + ' eos'
        target_text_input = 'sos ' + translation

        input_texts.append(input_text)
        target_texts.append(target_text)
        target_texts_inputs.append(target_text_input)

    print("num samples:", len(input_texts))

    def tokenizer(fit_text, filters=None):
        if filters:
            t = Tokenizer(num_words=config["basic"]["max_vocab_sz"], filters=filters)
        else:
            t = Tokenizer(num_words=config["basic"]["max_vocab_sz"])
        t.fit_on_texts(fit_text)
        return t

    # tokenize the inputs and get the word to index mapping for input language
    tokenizer_inputs = tokenizer(input_texts)
    input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
    word2idx_inputs = tokenizer_inputs.word_index
    print('Found %s unique input tokens.' % len(word2idx_inputs))

    max_len_input = max(len(s) for s in input_sequences)

    tokenizer_outputs = tokenizer(target_texts + target_texts_inputs, filters='')
    target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
    target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)
    word2idx_outputs = tokenizer_outputs.word_index
    print('Found %s unique output tokens.' % len(word2idx_outputs))

    assert ('sos' in word2idx_outputs)
    assert ('eos' in word2idx_outputs)

    max_len_target = max(len(s) for s in target_sequences)

    def make_embedd(embedd_file, word2idx, num_words, embed_dim):
        # make dict with pretrained word => vector consistency
        print('Loading word vectors...')
        word2vec = {}
        with open(embedd_file, 'r', encoding='utf8') as f:
            for line in f:
                values = line.split()
                try:
                    word2vec[values[0]] = np.asarray(values[1:], dtype='float32')
                except ValueError:
                    continue
        print('Found word vectors:', len(word2vec))

        # create embedding matrix
        print('Filling pre-trained embeddings...')
        n_words = min(num_words, len(word2idx) + 1)
        embedding_matrix = np.zeros((n_words, embed_dim))
        for word, i in word2idx.items():
            if i < num_words:
                embedding_vector = word2vec.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all zeros.
                    embedding_matrix[i] = embedding_vector

        return embedding_matrix, n_words

    embedding_matrix, embedding_matrix_ru = None, None
    if embedd_file_1:
        embedding_matrix, input_n_words = \
            make_embedd(embedd_file_1, word2idx_inputs, config["basic"]["max_vocab_sz"], config["basic"]["en_embed_dim"])
    if embedd_file_2:
        embedding_matrix_ru, output_n_words = \
            make_embedd(embedd_file_2, word2idx_outputs, config["basic"]["max_vocab_sz"], config["basic"]["ru_embed_dim"])

    config["basic"]["en_max_sq_len"] = max_len_input
    config["basic"]["ru_max_sq_len"] = max_len_target

    config["basic"]["en_max_vocab_sz"] = input_n_words
    config["basic"]["ru_max_vocab_sz"] = output_n_words

    for layer in config["model"]["layers"]:
        if layer["var_name"][0] == "encoder_embedding":
            layer["arguments"]["input_dim"] = config["basic"]["en_max_vocab_sz"]
            if embedd_file_1:
                layer["arguments"]["output_dim"] = embedding_matrix.shape[1]
        if layer["var_name"][0] == "decoder_embedding":
            layer["arguments"]["input_dim"] = config["basic"]["ru_max_vocab_sz"]
            if embedd_file_2:
                layer["arguments"]["output_dim"] = embedding_matrix_ru.shape[1]
        if layer["var_name"][0] == "output":
            layer["arguments"]["units"] = config["basic"]["ru_max_vocab_sz"]

    return input_sequences, \
           target_sequences_inputs, \
           target_sequences, \
           embedding_matrix, \
           embedding_matrix_ru, \
           word2idx_inputs, \
           word2idx_outputs, \
           config