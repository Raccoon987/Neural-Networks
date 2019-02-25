import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

import builtins



class KerasModel():

    def __init__(self):
        self.model = keras.models.Model

    def get_model(self):
        return self.model

    def set_model(self, inputs, outputs):
        self.model = self.model(inputs=inputs, outputs=outputs)

    def save_model(self, file_name):
        """
        save representation of the model as a JSON string and weights of the model as a HDF5 file
        :param file_name: path of file for model saving without file extension
        """

        model_json = self.model.to_json()
        with open(file_name + ".json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(file_name + ".h5")


    def compile_model(self, configs):
        '''

            :param configs: config file
            :return: None
        '''

        optimizers = {"SGD": keras.optimizers.SGD,
                      "RMSprop": keras.optimizers.RMSprop,
                      "Adagrad": keras.optimizers.Adagrad,
                      "Adadelta": keras.optimizers.Adadelta,
                      "Adam": keras.optimizers.Adam,
                      "Adamax": keras.optimizers.Adamax,
                      "Nadam": keras.optimizers.Nadam}

        # params = configs

        # e.g. "SGD" => keras.optimizers.SGD(**optimizer_args)
        configs["model"]["compile"]["optimizer"] = optimizers[configs["model"]["compile"]["optimizer"]]( \
            **configs["model"]["compile"]["optimizer_args"])
        configs["model"]["compile"].pop("optimizer_args", None)

        self.model.compile(**configs["model"]["compile"])


    def model_initializer(self, configs, embedding_mtx=None):
        '''
            :param embedding_mtx: dict key: embedding layer name, value: pretrained word=>vector embedding matrix
            :return: inputs, outputs layers to you can instantiate keras model
        '''

        layers_dict = {}  # OrderedDict()
        for input in configs["model"]["inputs"]:
            if "arguments" in input:
                if input["arguments"]["shape"] == "None":
                    input["arguments"]["shape"] = None
                input["arguments"]["shape"] = (input["arguments"]["shape"],)  # shape = (int_num, )
                layers_dict[input["var_name"]] = eval(input["type"])(**input["arguments"])
            else:
                layers_dict[input["var_name"]] = eval(input["type"])

        for layer in configs['model']['layers']:
            if layer["type"] == "keras.layers.Embedding":
                emb_name = layer["var_name"][0]
                if embedding_mtx:
                    if emb_name in embedding_mtx.keys():
                        layer["arguments"]["weights"] = [embedding_mtx[emb_name]]

            input = [layers_dict[layer["input"]]]
            if "initial_state" in layer.keys():
                # add initial states
                input.append([layers_dict[i] for i in layer["initial_state"]])

            if ("layer_wrapper" in layer.keys()):
                ''' layer wrappers case '''
                #Bidirectional(LSTM(layer_args), wrapper_args))(prev_layer_output, init_states)
                prelim_output = eval(layer["layer_wrapper"])(
                                                             eval(layer["type"])(
                                                                                 **layer["arguments"]
                                                                                ),
                                                             **layer["layer_wrapper_args"],
                                                             )(*input)
            else:
                if "arguments" in layer:
                    #LSTM(layer_args)(prev_layer_output)
                    prelim_output = eval(layer["type"])(
                                                        **layer["arguments"]
                                                        )(*input)
                else:
                    #import layer from athones model
                    prelim_output = eval(layer["type"])(*input)

            #add new layer to dictionary
            if isinstance(prelim_output, list):
                for idx, name in enumerate(layer["var_name"]):
                    layers_dict[name] = prelim_output[idx]
            else:
                layers_dict[layer["var_name"][0]] = prelim_output

        return [layers_dict[i] for i in configs["model"]["model_input"]], [layers_dict[o] for o in
                                                                           configs["model"]["model_output"]]


    def fit(self, origin, translation, targets, configs, show=True):
        '''

        :param origin: list of tokenized sentences in origin language
        :param translation: list of traslasted tokenized sentences
        :param targets: list of traslasted tokenized sentences shifted by 1 word
        :param configs: config file
        :param show:
        :return:
        '''

        batch_sz = configs["basic"]["batch_sz"]
        epochs = configs["basic"]["epochs"]
        validation_split = configs["basic"]["valid_split"]
        save_path = configs["model"]["save_dir"]
        num_words = configs["basic"]["ru_max_vocab_sz"]

        orig_train, orig_test, trans_train, trans_test, targ_train, targ_test = \
            train_test_split(origin, translation, targets, test_size=validation_split, random_state=42)

        train_n_batch = (len(orig_train) + batch_sz - 1) // batch_sz
        test_n_batch = (len(orig_test) + batch_sz - 1) // batch_sz

        def generator(x, y, tt):
            origin_max_len = configs["basic"]["en_max_sq_len"]
            trans_max_len = configs["basic"]["ru_max_sq_len"]

            n_batch = (len(x) + batch_sz - 1) // batch_sz

            while True:
                for batch in range(n_batch):
                    x_batch = x[batch * batch_sz: (batch + 1) * batch_sz]
                    y_batch = y[batch * batch_sz: (batch + 1) * batch_sz]
                    t_batch = tt[batch * batch_sz: (batch + 1) * batch_sz]

                    x_batch_len = max(len(s) for s in x_batch)
                    y_batch_len = max(len(s) for s in y_batch)
                    t_batch_len = max(len(s) for s in t_batch)

                    x_batch = keras.preprocessing.sequence.pad_sequences(x_batch, maxlen=x_batch_len)
                    y_batch = keras.preprocessing.sequence.pad_sequences(y_batch, maxlen=y_batch_len, padding='post')
                    t_batch = keras.preprocessing.sequence.pad_sequences(t_batch, maxlen=t_batch_len, padding='post')

                    one_hot_target = np.zeros((len(t_batch),
                                               t_batch_len,
                                               num_words),
                                              dtype='float32'
                                              )
                    for i, d in enumerate(t_batch):
                        for t, word in enumerate(d):
                            one_hot_target[i, t, word] = 1

                    yield [x_batch, y_batch], one_hot_target

        try:
            callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(epochs / 5)),
                         keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)]

            history = self.model.fit_generator(generator(orig_train, trans_train, targ_train),
                                               steps_per_epoch=train_n_batch,
                                               epochs=epochs,
                                               callbacks=callbacks,
                                               validation_data=generator(orig_test, trans_test, targ_test),
                                               validation_steps=test_n_batch)

            if show:
                plt.figure(figsize=(12, 9))  # make separate figure
                plt.subplot(2, 1, 1)
                plt.plot(history.history['loss'], label='loss')
                plt.plot(history.history['val_loss'], label='val_loss')
                plt.grid()
                plt.legend()

                plt.subplot(2, 1, 2)
                plt.plot(history.history['acc'], label='acc')
                plt.plot(history.history['val_acc'], label='val_acc')
                plt.legend()
                plt.grid()
                plt.show()

        except KeyboardInterrupt:
            print('KeyboardInterrupt: press control-c again to quit')
            return self.model.history['loss']


    def predict(self, data):
        return self.model.predict(data)




def load_model(filename):
    '''

    :param filename: name of files with json model structure and weights without file extension
    :return: model
    '''
    with open(filename + ".json", 'r') as json_model:
        loaded_model = keras.models.model_from_json(json_model.read())
        loaded_model.load_weights(filename + ".h5")

    return loaded_model