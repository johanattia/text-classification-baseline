""""Text classification using Convolutional Neural Networks"""

from typing import Callable, Dict, Union
import tensorflow as tf

from .text_model import TextClassificationModel
from ..layers.conv import conv_block
from ..utils import conv_utils


class CharacterConvNetv2(TextClassificationModel):
    def __init__(
        self,
        include_top: bool,
        n_classes: int = None,
        output_sequence_length: int = None,
        char_embedding: bool = False,
        embed_dim: int = None,
        dropout: float = 0.5,
        backbone_size: str = "small",
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
        super().__init__(
            include_top,
            n_classes,
            kernel_initializer,
            bias_initializer,
            embeddings_initializer,
            kernel_regularizer,
            bias_regularizer,
            embeddings_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            embeddings_constraint,
            **kwargs,
        )

    def _build_feature_encoder(self):
        raise NotImplementedError(
            "`_build_feature_encoder` must be implemented in child class."
        )

    def _build_classification_head(self):
        raise NotImplementedError(
            "`_build_classifier` must be implemented in child class."
        )

    def _process_text_input(self, inputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError(
            "`_process_text_input` must be implemented in child class."
        )

    def build_from_text_dataset(
        self, dataset: tf.data.Dataset, as_supervised: bool = True
    ):
        raise NotImplementedError(
            "`build_from_text_dataset` must be implemented in child class."
        )


class CharacterConvNet(tf.keras.Model):
    """Character-level Convolutional Neural Networks for Text Classification

    Reference:
    * `Character-level Convolutional Networks for Text Classification` (https://arxiv.org/pdf/1509.01626.pdf)

    ```python
    character_cnn = CharacterConvNet(
        include_top=True,
        n_classes=10,
        dropout=0.5,
        output_sequence_length=128
    )
    character_cnn.build_from_text_dataset(dataset, as_supervised=True)

    character_cnn.compile(optimizer="adam", loss="categorical_crossentropy")
    history = character_cnn.fit(dataset, epochs=20)
    ```

    Args:
        include_top (bool): _description_
        n_classes (int): _description_
        output_sequence_length (int): _description_
        char_embedding (bool, optional): _description_.
            Defaults to False.
        embed_dim (int, optional): _description_.
            Defaults to None.
        dropout (float, optional): _description_.
            Defaults to 0.4.
        backbone_size (str, optional): _description_.
            Defaults to "small".
        kernel_initializer (Union[str, Callable], optional): _description_.
            Defaults to "glorot_uniform".
        bias_initializer (Union[str, Callable], optional): _description_.
            Defaults to "zeros".
        embeddings_initializer (Union[str, Callable], optional): _description_.
            Defaults to "uniform".
        kernel_regularizer (Union[str, Callable], optional): _description_.
            Defaults to None.
        bias_regularizer (Union[str, Callable], optional): _description_.
            Defaults to None.
        embeddings_regularizer (Union[str, Callable], optional): _description_.
            Defaults to None.
        activity_regularizer (Union[str, Callable], optional): _description_.
            Defaults to None.
        kernel_constraint (Union[str, Callable], optional): _description_.
            Defaults to None.
        bias_constraint (Union[str, Callable], optional): _description_.
            Defaults to None.
        embeddings_constraint (Union[str, Callable], optional): _description_.
            Defaults to None.
    """

    def __init__(
        self,
        include_top: bool,
        n_classes: int,
        output_sequence_length: int,
        char_embedding: bool = False,
        embed_dim: int = None,
        dropout: float = 0.5,
        backbone_size: str = "small",
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
        self.output_sequence_length = output_sequence_length

        char_embedding, embed_dim = conv_utils.check_character_embedding(
            char_embedding, embed_dim
        )
        self._char_embedding = char_embedding
        self.embed_dim = embed_dim

        self.dropout_ = dropout
        self.backbone_size = conv_utils.normalize_backbone_size(backbone_size)

        # TEXT VECTORIZATION
        if self._char_embedding:
            self.char_vectorizer = tf.keras.layers.TextVectorization(
                standardize="lower",
                split="character",
                output_mode="int",
                output_sequence_length=self.output_sequence_length,
            )
        else:
            self.char_vectorizer = tf.keras.layers.TextVectorization(
                standardize="lower", split="character", output_mode="multi_hot"
            )

        # MODEL WEIGHTS PARAMETERS
        self._weights_parameters = dict(
            kernel_initializer=tf.keras.initializers.get(kernel_initializer),
            bias_initializer=tf.keras.initializers.get(bias_initializer),
            kernel_regularizer=tf.keras.regularizers.get(kernel_regularizer),
            bias_regularizer=tf.keras.regularizers.get(bias_regularizer),
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
            kernel_constraint=tf.keras.constraints.get(kernel_constraint),
            bias_constraint=tf.keras.constraints.get(bias_constraint),
        )
        self._embedding_parameters = dict(
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint,
        )

        self._is_built_from_texts = None

    def build(self, input_shape: tf.TensorShape):

        # CONV ENCODER
        self.conv_encoder = self._build_conv_encoder()

        # MLP CLASSIFIER
        if self._include_top:
            units_ = 1024 if self.backbone_size == "small" else 2048
            self.mlp_classifier = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        units=units_,
                        activation=tf.nn.relu,
                        use_bias=True,
                        **self._weights_parameters,
                    ),
                    tf.keras.layers.Dropout(rate=self.dropout_),
                    tf.keras.layers.Dense(
                        units=units_,
                        activation=tf.nn.relu,
                        use_bias=True,
                        **self._weights_parameters,
                    ),
                    tf.keras.layers.Dropout(rate=self.dropout_),
                    tf.keras.layers.Dense(
                        units=self.n_classes,
                        activation=None,
                        use_bias=True,
                        **self._weights_parameters,
                    ),
                ]
            )
        else:
            self.mlp_classifier = None

        super().build(input_shape)

    def _build_conv_encoder(self, name: str = "conv_encoder") -> tf.keras.Model:
        """_summary_

        Args:
            name (str, optional): _description_. Defaults to "conv_encoder".

        Returns:
            tf.keras.Model: _description_
        """
        filters_ = 256 if self.backbone_size == "small" else 1024

        if self._char_embedding:
            inputs = tf.keras.Input(shape=(self.output_sequence_length, self.embed_dim))
        else:
            inputs = tf.keras.Input(
                shape=(
                    self.output_sequence_length,
                    self.char_vectorizer.vocabulary_size(),
                )
            )

        x = conv_block(inputs, filters=filters_, kernel_size=7, block_id=1, pool_size=3)
        x = conv_block(x, filters=filters_, kernel_size=7, block_id=2, pool_size=3)
        x = conv_block(x, filters=filters_, kernel_size=3, block_id=3)
        x = conv_block(x, filters=filters_, kernel_size=3, block_id=4)
        x = conv_block(x, filters=filters_, kernel_size=3, block_id=5)
        x = conv_block(x, filters=filters_, kernel_size=3, block_id=6, pool_size=3)

        return tf.keras.Model(inputs=inputs, outputs=x, name=name)

    def build_from_text_dataset(
        self, dataset: tf.data.Dataset, as_supervised: bool = True
    ) -> tf.data.Dataset:
        """_summary_

        Args:
            dataset (tf.data.Dataset): _description_
            as_supervised (bool, optional): _description_. Defaults to True.

        Returns:
            tf.data.Dataset: _description_
        """
        if as_supervised:
            dataset = dataset.map(lambda x, _: x)

        self.character_vectorizer.adapt(dataset)

        if self._char_embedding:
            self.char_embedding = tf.keras.layers.Embedding(
                input_dim=self.character_vectorizer.vocabulary_size(),
                output_dim=self.embed_dim,
                **self._embedding_parameters,
            )

        self._is_built_from_texts = True

    def call(self, inputs: tf.Tensor, training: bool = None):
        # Vectorization: (B,) -> (B, L)
        indices = self.char_vectorizer(inputs)

        # (B, L) -> (B, L, F1)
        if self._char_embedding:
            indices = self.char_embedding(indices)
        else:
            indices = tf.one_hot(
                indices=indices, depth=self.char_vectorizer.vocabulary_size()
            )

        # Features: (B, L, F2)
        features = self.conv_encoder(indices)

        if self._include_top:
            return self.mlp_classifier(features, training=training)

        return features

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "include_top": self._include_top,
                "n_classes": self.n_classes,
                "dropout": self.dropout_,
                "output_sequence_length": self.output_sequence_length,
                "char_embedding": self._char_embedding,
                "embed_dim": self.embed_dim,
                "backbone_size": self.backbone_size,
                "kernel_initializer": tf.keras.initializers.serialize(
                    self._weights_parameters["kernel_initializer"]
                ),
                "bias_initializer": tf.keras.initializers.serialize(
                    self._weights_parameters["bias_initializer"]
                ),
                "kernel_regularizer": self.tf.keras.regularizers.serialize(
                    self._weights_parameters["kernel_regularizer"]
                ),
                "bias_regularizer": self.tf.keras.regularizers.serialize(
                    self._weights_parameters["bias_regularizer"]
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
            }
        )
        return config


class TextConvNet(tf.keras.Model):
    """Convolutional Neural Networks for Text Classification

    Reference:
    * `A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional
    Neural Networks for Sentence Classification` (https://arxiv.org/pdf/1510.03820.pdf)

    Args:
        tf (_type_): _description_
    """

    def __init__(self, n_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_classes = n_classes

    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training, mask)

    def get_config(self):
        return super().get_config()
