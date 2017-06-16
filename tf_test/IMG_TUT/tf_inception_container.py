from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import sys
import base64

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath('%s/../../../tf_models/slim/' % cur_dir))
sys.path.insert(0, os.path.abspath('%s/../../containers/python' % cur_dir))

import rpc
from nets import inception_v3
from preprocessing import inception_preprocessing
from datasets import imagenet

image_size = inception_v3.inception_v3.default_image_size
slim = tf.contrib.slim


class TfInceptionContainer(rpc.ModelContainerBase):

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path

    def predict_bytes(self, inputs):
        outputs = []
        with tf.Graph().as_default():
            decoded_images = [tf.image.decode_jpeg(input_image.tobytes(), channels=3) for input_image in inputs]
            preprocessed_images = [
                inception_preprocessing.preprocess_image(
                    decoded_image,
                    image_size,
                    image_size,
                    is_training=False) for decoded_image in decoded_images]

            with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                logits, _ = inception_v3.inception_v3(
                    preprocessed_images, num_classes=1001, is_training=False)

            all_probabilities = tf.nn.softmax(logits)
            init_fn = slim.assign_from_checkpoint_fn(
                self.checkpoint_path, slim.get_model_variables("InceptionV3"))

            with tf.Session() as sess:
                init_fn(sess)
                decoded_images, all_probabilities = sess.run(
                    [decoded_images, all_probabilities])

                for input_probabilities in all_probabilities:
                    sorted_inds = [i[0] for i in sorted(
                        enumerate(-input_probabilities), key=lambda x:x[1])]
                    outputs.append(str(sorted_inds[0]))

                    names = imagenet.create_readable_names_for_imagenet_labels()
                    for i in range(5):
                        index = sorted_inds[i]
                        print('Probability %0.2f%% => [%s]' % (input_probabilities[index] * 100, names[index]))

        return outputs


if __name__ == "__main__":
    print("Starting InceptionV3 Cifar container")
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

    input_type = "bytes"
    model_checkpoint_path = os.environ["CLIPPER_MODEL_CHECKPOINT_PATH"]
    model = TfInceptionContainer(model_checkpoint_path)
    rpc_service = rpc.RPCService()
    rpc_service.start(model, ip, port, model_name, model_version, input_type)
