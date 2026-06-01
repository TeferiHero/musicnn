import os
import numpy as np
import librosa
import keras
import tensorflow as tf
import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
cluster_weights = tfmot.clustering.keras.cluster_weights

from musicnn.v2 import models
from musicnn.v2 import configuration as config


from musicnn.v2.extractor import extractor
from musicnn.v2.tagger import top_tags
from musicnn.v2.dataset import get_dataset

import random
random.seed(42)


def cluster():
    model = tf.keras.saving.load_model("stripped_pruned_model.keras")
    sparsity_clustered_model = tf.keras.models.clone_model(model, clone_function=apply_clustering)
    sparsity_clustered_model.summary()
    
    sparsity_clustered_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

    train_ds, val_ds, test_ds = get_dataset(augument_training=True, all_noise_ratio=0.25)

    sparsity_clustered_model.fit(train_ds, epochs=2)

    stripped_clustered_model = tfmot.clustering.keras.strip_clustering(sparsity_clustered_model)

    stripped_clustered_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
    
    print(stripped_clustered_model.evaluate(test_ds))
    stripped_clustered_model.save("stripped_clustered_model.keras")


def apply_clustering(layer):
    clustering_params = {
        'number_of_clusters': 16,
        'cluster_centroids_init': CentroidInitialization.KMEANS_PLUS_PLUS,
        'preserve_sparsity': True
    }
    
    if isinstance(layer, tf.keras.layers.Conv2D) \
       or isinstance(layer, tf.keras.layers.Dense) and layer.activity_regularizer is None:
        return cluster_weights(layer, **clustering_params)

    return layer

    
if __name__ == '__main__':
    cluster()
