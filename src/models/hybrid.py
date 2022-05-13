"""Stacked model for text classification: 
* Word-level Bidirectional GRU
* Character-level CNN: https://arxiv.org/pdf/1509.01626.pdf
* TF-IDF/PCA
* TF-IDF/Non-linear Dimensionality Reduction

On top : attention (https://arxiv.org/pdf/1703.03130.pdf) + Variable Selection Networks
"""

import tensorflow as tf


class HybridClassificationModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training, mask)

    def get_config(self):
        return super().get_config()
