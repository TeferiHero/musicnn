import os
import numpy as np
import librosa
import keras
import tensorflow as tf
import tensorflow_model_optimization as tfmot
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

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

    pruned_model = tf.keras.models.clone_model(model, clone_function=apply_pruning)

    pruned_model.summary()
    tf.keras.utils.plot_model(pruned_model, show_shapes=True, to_file="pruned.png")
    return

    pruned_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep()
    ]

    train_ds, val_ds, test_ds = get_dataset(augument_training=True, all_noise_ratio=0.25)
    pruned_model.fit(
        train_ds,
        epochs=3,
        callbacks=callbacks
    )
    
    stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    stripped_pruned_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
    
    print(stripped_pruned_model.evaluate(test_ds))
    stripped_pruned_model.save("stripped_pruned_model.keras")


def apply_pruning(layer):
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.3, begin_step=0, frequency=100)
    }
    
    if isinstance(layer, tf.keras.layers.Conv2D) \
       or isinstance(layer, tf.keras.layers.Dense) and layer.activity_regularizer is None:
        return prune_low_magnitude(layer, **pruning_params)

    return layer


def unfreeze(model):
    model.trainable = True
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


if __name__ == '__main__':
    prune()
