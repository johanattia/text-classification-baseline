"""Abstract model for text classification"""


from abc import abstractmethod
from typing import Callable, Dict, Union

import tensorflow as tf


class TextClassificationModel(tf.keras.Model):
    """Abstract model for text classification.

    Methods to be overriden in child class:
    * _build_feature_encoder
    * _build_classification_head
    * _process_text_input
    * build_from_text_dataset
    """

    def __init__(
        self,
        include_top: bool,
        n_classes: int = None,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        bias_initializer: Union[str, Callable] = "zeros",
        embeddings_initializer: Union[str, Callable] = "uniform",
        kernel_regularizer: Union[str, Callable] = None,
        bias_regularizer: Union[str, Callable] = None,
        embeddings_regularizer: Union[str, Callable] = None,
        activity_regularizer: Union[str, Callable] = None,
        kernel_constraint: Union[str, Callable] = None,
        bias_constraint: Union[str, Callable] = None,
        embeddings_constraint: Union[str, Callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._include_top = include_top
        self.n_classes = n_classes

        # Model weights parameters
        self._weights_parameters = dict(
            kernel_initializer=tf.keras.initializers.get(kernel_initializer),
            bias_initializer=tf.keras.initializers.get(bias_initializer),
            kernel_regularizer=tf.keras.regularizers.get(kernel_regularizer),
            bias_regularizer=tf.keras.regularizers.get(bias_regularizer),
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
            kernel_constraint=tf.keras.constraints.get(kernel_constraint),
            bias_constraint=tf.keras.constraints.get(bias_constraint),
        )
        self._embeddings_parameters = dict(
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint,
        )

        self._is_built_from_texts = None

    def build(self, input_shape: tf.TensorShape):
        # Feature encoder
        self.feature_encoder = self._build_feature_encoder()

        # Classification head
        self.classification_head = None
        if self._include_top:
            self.classification_head = self._build_classification_head()

        super().build(input_shape)

    @abstractmethod
    def _build_feature_encoder(self):
        raise NotImplementedError(
            "`_build_feature_encoder` must be implemented in child class."
        )

    @abstractmethod
    def _build_classification_head(self):
        raise NotImplementedError(
            "`_build_classifier` must be implemented in child class."
        )

    @abstractmethod
    def _process_text_input(self, inputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError(
            "`_process_text_input` must be implemented in child class."
        )

    @abstractmethod
    def build_from_text_dataset(
        self, dataset: tf.data.Dataset, as_supervised: bool = True
    ):
        raise NotImplementedError(
            "`build_from_text_dataset` must be implemented in child class."
        )

    def call(self, inputs: tf.Tensor, training: bool = None):
        # Vectorization of a batch of string tensors: (B,) -> (B, L, F)
        inputs, mask = self._process_text_input(inputs)

        # Feature encoding: (B, L, F) -> (B, D)
        if mask is not None:
            features = self.feature_encoder(inputs, training=training, mask=mask)
        else:
            features = self.feature_encoder(inputs, training=training)

        # Classification: (B, D) -> (B, N_CLASSES)
        if self._include_top:
            return self.classification_head(features, training=training)

        return features

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "include_top": self._include_top,
                "n_classes": self.n_classes,
                "kernel_initializer": tf.keras.initializers.serialize(
                    self._weights_parameters["kernel_initializer"]
                ),
                "bias_initializer": tf.keras.initializers.serialize(
                    self._weights_parameters["bias_initializer"]
                ),
                "embeddings_initializer": tf.keras.initializers.serialize(
                    self._embeddings_parameters["embeddings_initializer"]
                ),
                "kernel_regularizer": self.tf.keras.regularizers.serialize(
                    self._weights_parameters["kernel_regularizer"]
                ),
                "bias_regularizer": self.tf.keras.regularizers.serialize(
                    self._weights_parameters["bias_regularizer"]
                ),
                "embeddings_regularizer": tf.keras.regularizers.serialize(
                    self._embeddings_parameters["embeddings_regularizer"]
                ),
                "activity_regularizer": self.tf.keras.regularizers.serialize(
                    self._weights_parameters["activity_regularizer"]
                ),
                "kernel_constraint": self.tf.keras.constraints.serialize(
                    self._weights_parameters["kernel_constraint"]
                ),
                "bias_constraint": self.tf.keras.constraints.serialize(
                    self._weights_parameters["bias_constraint"]
                ),
                "embeddings_constraint": tf.keras.constraints.serialize(
                    self._embeddings_parameters["embeddings_constraint"]
                ),
            }
        )
        return config
