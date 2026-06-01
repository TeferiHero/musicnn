import os
import numpy as np
import librosa
import keras
import tensorflow as tf

from musicnn.v2 import models
from musicnn.v2 import configuration as config


from musicnn.v2.extractor import extractor
from musicnn.v2.tagger import top_tags
from musicnn.v2.dataset import get_dataset

import random
random.seed(42)


def prune():
    model = tf.keras.saving.load_model("modelv2-pure.keras")
    unfreeze(model)

    model.summary()


def unfreeze(model):
    model.trainable = True
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


if __name__ == '__main__':
    prune()
