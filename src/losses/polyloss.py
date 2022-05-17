"PolyLoss for classification"


import tensorflow as tf


class PolyLoss(tf.keras.losses.Loss):
    """_summary_

    Reference:
    * `PolyLoss: A Polynomial Expansion Perspective of Classification Loss
    Functions` (https://arxiv.org/pdf/2204.12511.pdf)
    """

    def __init__(self, reduction=..., name=None):
        super().__init__(reduction, name)

    def call(self, y_true, y_pred):
        return super().call(y_true, y_pred)
