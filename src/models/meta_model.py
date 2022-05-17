"""Stacked model for text classification: 
* Word-level Bidirectional GRU
* Character-level CNN: https://arxiv.org/pdf/1509.01626.pdf
* TF-IDF/PCA
* TF-IDF/Non-linear Dimensionality Reduction

On top : attention (https://arxiv.org/pdf/1703.03130.pdf) + Variable Selection Networks
"""

from typing import Dict
import tensorflow as tf

from ..layers.feedforward import FeedForwardNetwork


class MetaClassificationModel(tf.keras.Model):
    def __init__(
        self,
        models: Dict[str, tf.keras.Model],
        aggregation_level: str = "feature",
        **kwargs
    ):
        super().__init__(**kwargs)

        if str(aggregation_level).lower() not in ["feature", "stacking", "ensembling"]:
            raise ValueError(
                "`aggregation_level` must be in [`feature`, `stacking`, `ensembling`]"
            )
        self._aggregation_level = str(aggregation_level).lower()

        self.models = models
        self._is_initialized_from_texts = None

    def initialize_from_text_dataset(
        self, dataset: tf.data.Dataset, as_supervised: bool = True
    ):
        raise NotImplementedError

    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training, mask)

    def get_config(self):
        return super().get_config()
