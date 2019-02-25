from translator_preprocess import preprocess
from translator_model import KerasModel, load_model

import os, sys
import json
import numpy as np
import pickle
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

import builtins


if __name__ == "__main__":

    f_path = lambda f_name: os.path.realpath(os.path.join(os.getcwd(), f_name)).replace("\\", "/")

    config = json.load(open(f_path('translator_train.json'), 'r'))

    '''
    input_sequences, \
    target_sequences_inputs, \
    target_sequences, \
    embedding_matrix, \
    embedding_matrix_ru, \
    word2idx_inputs, \
    word2idx_outputs, \
    config = preprocess('en_rus_translator.txt',
                        config,
                        embedd_file_1=f_path("glove.6B.100d.txt"),
                        embedd_file_2="C:/wiki.ru.vec")
    
    # SAVE
    np.save(f_path("translator/en_embedding100"), embedding_matrix)
    np.save(f_path("translator/rus_embedding300"), embedding_matrix_ru)
    with open(f_path("translator/en_sent_list") + '.pkl', "wb") as f:
        pickle.dump(input_sequences, f)
    with open(f_path("translator/rus_sent_list") + '.pkl', "wb") as f:
        pickle.dump(target_sequences_inputs, f)
    with open(f_path("translator/rus_sent_list_targ") + '.pkl', "wb") as f:
        pickle.dump(target_sequences, f)

    with open(f_path("translator/en_word2idx") + '.pkl', 'wb') as f:
        pickle.dump(word2idx_inputs, f, pickle.HIGHEST_PROTOCOL)
    with open(f_path("translator/rus_word2idx") + '.pkl', 'wb') as f:
        pickle.dump(word2idx_outputs, f, pickle.HIGHEST_PROTOCOL)

    with open(f_path("translator/translation_train_config.json"), 'w') as outfile:
        json.dump(config, outfile)
    '''

    #LOAD
    config = json.load(open(f_path("translator/translation_train_config.json"), 'r'))
    with open(f_path("translator/en_word2idx") + '.pkl', 'rb') as f:
        word2idx_inputs = pickle.load(f)
    with open(f_path("translator/rus_word2idx") + '.pkl', 'rb') as f:
        word2idx_outputs = pickle.load(f)

    embedding_matrix = np.load(f_path("translator/en_embedding100.npy"))
    embedding_matrix_ru = np.load(f_path("translator/rus_embedding300.npy"))

    with open(f_path("translator/en_sent_list") + '.pkl', 'rb') as f:
        input_sequences = pickle.load(f)
    with open(f_path("translator/rus_sent_list") + '.pkl', 'rb') as f:
        target_sequences_inputs = pickle.load(f)
    with open(f_path("translator/rus_sent_list_targ") + '.pkl', 'rb') as f:
        target_sequences = pickle.load(f)

    config["basic"]["epochs"] = 25
    config["model"]["compile"]["optimizer_args"]["lr"] = 0.001

    print(json.dumps(config, indent=4))
    '''
    m = KerasModel()
    input_lst, output_lst = m.model_initializer(configs=config, embedding_mtx={"encoder_embedding": embedding_matrix,
                                                                               "decoder_embedding": embedding_matrix_ru})
    m.set_model(input_lst, output_lst)
    m.compile_model(config)
    # m.fit(encoder_inputs, decoder_inputs, decoder_targets_one_hot, config)
    m.fit(input_sequences, target_sequences_inputs, target_sequences, config)
    m.save_model(f_path("translator/translator_model"))

    model = m.get_model()
    '''
    model = load_model(f_path("translator/translator_model"))

    builtins.model = model

    ##### Make predictions #####
    _, enc_h_s, enc_c_s = model.get_layer('enc_lstm').output
    encoder_m = Model(model.get_layer('encoder_input').output, [enc_h_s, enc_c_s])

    prediction_config = json.load(open(f_path('translator_predict.json'), 'r'))

    decoder_m = KerasModel()
    input_lst_d, output_lst_d = decoder_m.model_initializer(configs=prediction_config, import_model=model)
    decoder_m.set_model(input_lst_d, output_lst_d)

    idx2word_eng = {v: k for k, v in word2idx_inputs.items()}
    idx2word_trans = {v: k for k, v in word2idx_outputs.items()}


    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_m.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))

        # Populate the first character of target sequence with the start character.
        # NOTE: tokenizer lower-cases all words
        target_seq[0, 0] = word2idx_outputs['sos']

        # if we get this we break
        eos = word2idx_outputs['eos']

        # Create the translation
        output_sentence = []
        for _ in range(config["basic"]["ru_max_sq_len"]):
            output_tokens, h, c = decoder_m.predict(
                [target_seq] + states_value
            )
            # output_tokens, h = decoder_model.predict(
            #     [target_seq] + states_value
            # ) # gru

            # Get next word
            idx = np.argmax(output_tokens[0, 0, :])

            # End sentence of EOS
            if eos == idx:
                break

            word = ''
            if idx > 0:
                word = idx2word_trans[idx]
                output_sentence.append(word)

            # Update the decoder input
            # which is just the word just generated
            target_seq[0, 0] = idx

            # Update states
            states_value = [h, c]
            # states_value = [h] # gru

        return ' '.join(output_sentence)



    while True:
        # Do some test translations
        i = np.random.choice(len(input_sequences))
        input_seq = input_sequences[i:i + 1]
        input_seq = pad_sequences(input_seq, maxlen=config["basic"]["en_max_sq_len"])
        translation = decode_sequence(input_seq)
        print('-')
        print('Input:', input_seq, type(input_seq))
        print('Input:', [idx2word_eng[idx] for idx in input_seq[0] if idx != 0])
        print('Translation:', translation)

        ans = input("Continue? [Y/n]")
        if ans and ans.lower().startswith('n'):
            break