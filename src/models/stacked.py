"""Stacked model for text classification: 
* Word-level Bidirectional GRU
* Character-level CNN: https://arxiv.org/pdf/1509.01626.pdf
* TF-IDF/PCA
* TF-IDF/Non-linear Dimensionality Reduction
"""

import tensorflow as tf
