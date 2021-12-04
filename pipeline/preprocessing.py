import os
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import TextVectorization

from features import Feature


def _fill_in_missing(x):
    """Replace missing values in a SparseTensor.
    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
    Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
    Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
    """
    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value), axis=1)


def preprocessing_fn(inputs):
    """Preprocesses Dataset."""
    
    outputs = {}

    # Scale numerical features.
    for key in Feature.NUMERIC_FEATURE_KEYS:
        outputs[Feature.transformed_name(key)] = tft.scale_to_z_score(
            _fill_in_missing(inputs[key]))

    # Generate vocabularies and maps categorical features.
    for key in Feature.CATEGORICAL_FEATURE_KEYS:
        outputs[Feature.transformed_name(key)] = tft.compute_and_apply_vocabulary(
            x=_fill_in_missing(inputs[key]), num_oov_buckets=1, vocab_filename=key)

    # Convert Cover_Type to dense tensor.
    outputs[Feature.transformed_name(Feature.LABEL_KEY)] = _fill_in_missing(
        inputs[Feature.LABEL_KEY])

    return outputs
