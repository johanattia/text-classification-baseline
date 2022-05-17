"""Conv blocks/layers for text classification models"""


from typing import Callable, Union
import tensorflow as tf


def conv_block_v2(
    filters: int,
    kernel_size: int,
    block_id: int,
    activation: Union[str, Callable] = tf.nn.relu,
    pool_size: int = None,
    **kwargs,
):
    layers = [
        tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation=activation,
            name=f"conv{block_id}",
            **kwargs,
        )
    ]
    if pool_size is not None:
        layers += [
            tf.keras.layers.MaxPool1D(pool_size=pool_size, name=f"pooling{block_id}")
        ]
    return tf.keras.Sequential(layers)


def conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    block_id: int,
    pool_size: int = None,
    kernel_initializer: Union[str, Callable] = "glorot_uniform",
    bias_initializer: Union[str, Callable] = "zeros",
    kernel_regularizer: Union[str, Callable] = None,
    bias_regularizer: Union[str, Callable] = None,
    activity_regularizer: Union[str, Callable] = None,
    kernel_constraint: Union[str, Callable] = None,
    bias_constraint: Union[str, Callable] = None,
) -> tf.Tensor:
    """_summary_"""
    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.relu,
        name=f"conv{block_id}",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
    )(x)

    if pool_size is not None:
        x = tf.keras.layers.MaxPool1D(pool_size=pool_size, name=f"pooling{block_id}")(x)

    return x
