from __future__ import print_function
import sys
import os
import re
import numpy as np
import movie_lstm as mlstm
import imdb
import six.moves.cPickle as pickle

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath('%s/..' % cur_dir))

import rpc

PREDICTION_LABEL_POSITIVE = 1
PREDICTION_LABEL_NEGATIVE = 0

class TheanoLstmContainer(rpc.ModelContainerBase):

    def __init__(self, model_path, options_path, imdb_pkl_dict_path):
        print("Initializing container...")
        model_options_file = open(options_path, "r")
        model_options = {}
        for line in model_options_file.readlines():
            key, value = line.split('@')
            value = value.strip()
            try:
                value = int(value)
            except ValueError as e:
                pass
            model_options[key] = value

        self.n_words = model_options['n_words']
        params = mlstm.init_params(model_options)
        params = mlstm.load_params(model_path, params)
        tparams = mlstm.init_tparams(params)

        (use_noise, x, mask, y, f_pred_prob, f_pred, cost) = mlstm.build_model(tparams, model_options)
        self.predict_function = f_pred

        dict_file = open(imdb_pkl_dict_path, 'rb')
        self.imdb_dict = pickle.load(dict_file)
        print("Done!")

    def get_imdb_indices(self, input_str):
        split_input = input_str.split(" ")
        indices = np.ones(len(split_input))
        for i in range(0, len(split_input)):
            term = split_input[i]
            term = re.sub('[^a-zA-Z\d\s:]', '', term)
            if term in self.imdb_dict:
                index = self.imdb_dict[term]
                if index < self.n_words:
                    indices[i] = index
        return indices

    def predict_strings(self, inputs):
        outputs = []
        for input_str in inputs:
            indices = self.get_imdb_indices(input_str)
            x, mask, y = imdb.prepare_data([indices], [], maxlen=None)
            prediction = self.predict_function(x, mask)[0]
            outputs.append(str(prediction))
        return outputs

if __name__ == "__main__":
    print("Starting Sentiment Analysis Lstm Container")
    try:
        model_name = os.environ["CLIPPER_MODEL_NAME"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_NAME environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        model_version = os.environ["CLIPPER_MODEL_VERSION"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_VERSION environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
    	lstm_model_path = os.environ["CLIPPER_MODEL_PATH"]
    except KeyError:
    	print(
    		"ERROR: CLIPPER_MODEL_PATH environment variable must be set",
    		file=sys.stdout)
    	sys.exit(1)
    try:
    	model_options_path = os.environ["CLIPPER_MODEL_OPTIONS_PATH"]
    except KeyError:
    	print(
    		"ERROR: CLIPPER_MODEL_OPTIONS_PATH environment variable must be set",
    		file=sys.stdout)
    	sys.exit(1)
    try:
        imdb_pkl_dict_path = os.environ["CLIPPER_IMDB_DICT_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_IMDB_DICT_PATH environment variable must be set",
            file=sys.stdout)
        sys.exit(1)    

    ip = "127.0.0.1"
    if "CLIPPER_IP" in os.environ:
        ip = os.environ["CLIPPER_IP"]
    else:
        print("Connecting to Clipper on localhost")

    port = 7000
    if "CLIPPER_PORT" in os.environ:
        port = int(os.environ["CLIPPER_PORT"])
    else:
        print("Connecting to Clipper with default port: 7000")

    input_type = "strings"
    container = TheanoLstmContainer(lstm_model_path, model_options_path, imdb_pkl_dict_path)
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, port, model_name, model_version, input_type)

