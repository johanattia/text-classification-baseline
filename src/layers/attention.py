"""Inner Attention"""


from typing import Callable, Dict, Tuple, Union
import tensorflow as tf


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

    @classmethod
    def from_config(cls, config: Dict):
        config["kernel_initializer"] = tf.keras.initializers.deserialize(
            config["kernel_initializer"]
        )
        config["bias_initializer"] = tf.keras.initializers.deserialize(
            config["bias_initializer"]
        )
        config["kernel_regularizer"] = tf.keras.regularizers.deserialize(
            config["kernel_regularizer"]
        )
        config["bias_regularizer"] = tf.keras.regularizers.deserialize(
            config["bias_regularizer"]
        )
        config["activity_regularizer"] = tf.keras.regularizers.deserialize(
            config["activity_regularizer"]
        )
        config["kernel_constraint"] = tf.keras.constraints.deserialize(
            config["kernel_constraint"]
        )
        config["bias_constraint"] = tf.keras.constraints.deserialize(
            config["bias_constraint"]
        )
        return cls(**config)
