"""Single-hidden Layer Feed Forward Network"""


from typing import Callable, Iterable, Union
import itertools

import tensorflow as tf


def FeedForwardNetworkv2(
    output_units: int = None,
    hidden_units: Iterable[int] = [],
    output_activation: Union[str, Callable] = None,
    hidden_activation: Union[str, Callable] = None,
    use_bias: bool = True,
    dropout: float = 0.5,
    **kwargs,
):
    layers = list(
        itertools.chain.from_iterable(
            [
                [
                    tf.keras.layers.Dense(
                        units=units,
                        activation=hidden_activation,
                        use_bias=use_bias,
                        **kwargs,
                    ),
                    tf.keras.layers.Dropout(rate=dropout),
                ]
                for units in hidden_units
            ]
        )
    )
    if output_units is not None:
        layers += [
            tf.keras.layers.Dense(
                units=output_units,
                activation=output_activation,
                use_bias=use_bias,
                **kwargs,
            )
        ]
    return tf.keras.Sequential(layers)


def FeedForwardNetwork(
    hidden_dim: int,
    output_dim: int,
    dropout: float,
    use_bias: bool = True,
    hidden_activation: Union[str, Callable] = None,
    output_activation: Union[str, Callable] = None,
    kernel_initializer: Union[str, Callable] = "glorot_uniform",
    bias_initializer: Union[str, Callable] = "zeros",
    kernel_regularizer: Union[str, Callable] = None,
    bias_regularizer: Union[str, Callable] = None,
    activity_regularizer: Union[str, Callable] = None,
    kernel_constraint: Union[str, Callable] = None,
    bias_constraint: Union[str, Callable] = None,
    **kwargs,
) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                units=hidden_dim,
                activation=hidden_activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                **kwargs,
            ),
            tf.keras.layers.Dropout(rate=dropout),
            tf.keras.layers.Dense(
                units=output_dim,
                activation=output_activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                **kwargs,
            ),
            tf.keras.layers.Dropout(rate=dropout),
        ]
    )
