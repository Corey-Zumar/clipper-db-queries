#!/usr/bin/env bash

set -e
set -u
set -o pipefail

trap "exit" INT TERM
trap "kill 0" EXIT

checkpoint_path=$1

export CLIPPER_MODEL_NAME="imgnet_inception_model"
export CLIPPER_MODEL_VERSION="1"
export CLIPPER_MODEL_CHECKPOINT_PATH=$checkpoint_path

python ../../containers/python/tf_inception_container.py 
