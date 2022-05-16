"""Inner Attention"""


from typing import Callable, Dict, Iterable, Tuple, Union
import tensorflow as tf

from .feedforward import FeedForwardNetwork


class InnerAttention(tf.keras.layers.Layer):
    """_summary_

    Args:
        units (int): _description_
        normalize (bool): _description_
        kernel_initializer (Union[str, Callable], optional): _description_.
            Defaults to "glorot_uniform".
        bias_initializer (Union[str, Callable], optional): _description_.
            Defaults to "zeros".
        kernel_regularizer (Union[str, Callable], optional): _description_.
            Defaults to None.
        bias_regularizer (Union[str, Callable], optional): _description_.
            Defaults to None.
        activity_regularizer (Union[str, Callable], optional): _description_.
            Defaults to None.
        kernel_constraint (Union[str, Callable], optional): _description_.
            Defaults to None.
        bias_constraint (Union[str, Callable], optional): _description_.
            Defaults to None.
    """

    def __init__(
        self,
        units: int,
        normalize: bool,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        bias_initializer: Union[str, Callable] = "zeros",
        kernel_regularizer: Union[str, Callable] = None,
        bias_regularizer: Union[str, Callable] = None,
        activity_regularizer: Union[str, Callable] = None,
        kernel_constraint: Union[str, Callable] = None,
        bias_constraint: Union[str, Callable] = None,
        **kwargs,
    ):
        super(InnerAttention, self).__init__(**kwargs)

        self._units = units
        self._normalize = normalize

        self._weights_parameters = dict(
            kernel_initializer=tf.keras.initializers.get(kernel_initializer),
            bias_initializer=tf.keras.initializers.get(bias_initializer),
            kernel_regularizer=tf.keras.regularizers.get(kernel_regularizer),
            bias_regularizer=tf.keras.regularizers.get(bias_regularizer),
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
            kernel_constraint=tf.keras.constraints.get(kernel_constraint),
            bias_constraint=tf.keras.constraints.get(bias_constraint),
        )

    def build(self, input_shape):
        # Trainable context query to compute similarities/dot products
        # with sequence elements
        self.context_query = self.add_weight(
            name="context_query",
            shape=[self._units, 1],
            dtype=tf.float32,
            initializer=self._weights_parameters["kernel_initializer"],
            regularizer=self._weights_parameters["kernel_regularizer"],
            trainable=True,
            constraint=self._weights_parameters["kernel_constraint"],
        )

        self.softmax = tf.keras.layers.Softmax(axis=1)
        self.dense = tf.keras.layers.Dense(
            units=self._units,
            activation=tf.nn.relu,
            use_bias=True,
            **self._weights_parameters,
        )

        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        mask: tf.Tensor = None,
        return_attention_scores: bool = False,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor]]:
        # (B, L, F) -> (B, L, U)
        projections = self.dense(inputs)

        # (B, L, 1)
        if self._normalize:
            similarities = tf.matmul(
                tf.math.l2_normalize(projections, axis=-1),
                tf.math.l2_normalize(self.context_query),
            )
        else:
            similarities = tf.matmul(projections, self.context_query)

        # (B, L, 1)
        attention_scores = self.sotfmax(similarities, tf.cast(mask, dtype=tf.bool))

        # (B, L, F)
        output = attention_scores * inputs

        if return_attention_scores:
            return output, attention_scores

        return output

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "units": self._units,
                "normalize": self._normalize,
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


class TransformerBlock(tf.keras.layers.Layer):
    """_summary_

    Args:
        num_heads (int): _description_
        embed_dim (int): _description_
        hidden_dim (int): _description_
        ffn_output (bool, optional): _description_.
            Defaults to False.
        dropout (float, optional): _description_.
            Defaults to 0.1.
        epsilon (float, optional): _description_.
            Defaults to 1e-6.
        kernel_initializer (Union[str, Callable], optional): _description_.
            Defaults to "glorot_uniform".
        bias_initializer (Union[str, Callable], optional): _description_.
            Defaults to "zeros".
        kernel_regularizer (Union[str, Callable], optional): _description_.
            Defaults to None.
        bias_regularizer (Union[str, Callable], optional): _description_.
            Defaults to None.
        activity_regularizer (Union[str, Callable], optional): _description_.
            Defaults to None.
        kernel_constraint (Union[str, Callable], optional): _description_.
            Defaults to None.
        bias_constraint (Union[str, Callable], optional): _description_.
            Defaults to None.
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        hidden_dim: int,
        ffn_output: bool = False,
        dropout: float = 0.1,
        epsilon: float = 1e-6,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        bias_initializer: Union[str, Callable] = "zeros",
        kernel_regularizer: Union[str, Callable] = None,
        bias_regularizer: Union[str, Callable] = None,
        activity_regularizer: Union[str, Callable] = None,
        kernel_constraint: Union[str, Callable] = None,
        bias_constraint: Union[str, Callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.ffn_output = ffn_output
        self.dropout = dropout
        self.epsilon = epsilon

        self._weights_parameters = dict(
            kernel_initializer=tf.keras.initializers.get(kernel_initializer),
            bias_initializer=tf.keras.initializers.get(bias_initializer),
            kernel_regularizer=tf.keras.regularizers.get(kernel_regularizer),
            bias_regularizer=tf.keras.regularizers.get(bias_regularizer),
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
            kernel_constraint=tf.keras.constraints.get(kernel_constraint),
            bias_constraint=tf.keras.constraints.get(bias_constraint),
        )

    def build(self, input_shape: tf.TensorShape):
        # Defining Transformer block layers
        self.attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim,
            dropout=self.dropout,
            use_bias=True,
            **self._weights_parameters,
        )

        self.feed_forward_network = FeedForwardNetwork(
            hidden_dim=self.hidden_dim,
            output_dim=self.embed_dim,
            dropout=self.dropout,
            use_bias=True,
            hidden_activation=tf.nn.relu,
            output_activation=None,
            **self._weights_parameters,
        )

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)

        super().build(input_shape)

    def call(
        self, inputs: tf.Tensor, training: bool = False, mask: tf.Tensor = None
    ) -> tf.Tensor:
        """_summary_
        Args:
            inputs (tf.Tensor): _description_
            training (bool, optional): _description_. Defaults to False.
            mask (tf.Tensor, optional): _description_. Defaults to None.
        Returns:
            tf.Tensor: _description_
        """
        attention_output = self.compute_attention(
            inputs, training=training, attention_mask=mask
        )
        output1 = self.layer_norm1(inputs + attention_output)

        ffn_output = self.feed_forward_network(output1)
        output2 = self.layer_norm2(output1 + ffn_output)

        if self.ffn_output:
            return output2, ffn_output

        return output2

    def compute_attention(
        self, inputs: tf.Tensor, training: bool, attention_mask: tf.Tensor
    ) -> tf.Tensor:
        return self.attention_layer(
            inputs, inputs, training=training, attention_mask=attention_mask
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "embed_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim,
                "ffn_output": self.ffn_output,
                "dropout": self.dropout,
                "epsilon": self.epsilon,
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
