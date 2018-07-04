import os

import tensorflow as tf


def reduce_logging_output():
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
