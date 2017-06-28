set -e
set -u
set -o pipefail

trap "exit" INT TERM
trap "kill 0" EXIT

model_path=$1
options_path=$2
imdb_dict_path=$3

export CLIPPER_MODEL_NAME="imdb_lstm_model"
export CLIPPER_MODEL_VERSION="1"
export CLIPPER_MODEL_PATH=$model_path
export CLIPPER_MODEL_OPTIONS_PATH=$options_path
export CLIPPER_IMDB_DICT_PATH=$imdb_dict_path

python ../../containers/python/theano_lstm/theano_lstm_container.py
