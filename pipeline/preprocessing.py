import tensorflow as tf
from tensorflow.keras.preprocessing import sequence

from features import Feature

MAX_SEQUENCE_LENGTH = 500

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


def preprocessing_fn(inputs, tokenizer):
    """Preprocesses Dataset."""

    outputs = {}

    key = Feature.transformed_name(Feature.FEATURE_KEY)
    # Fill in missing values and create dense tensors.
    outputs[key] = _fill_in_missing(inputs[key])

    outputs[key] = tokenizer.texts_to_sequences(list(outputs[key]))
    outputs[key] = sequence.pad_sequences(outputs[key], maxlen=MAX_SEQUENCE_LENGTH)

    # Convert label to dense tensor.
    outputs[Feature.transformed_name(Feature.LABEL_KEY)] = _fill_in_missing(inputs[Feature.LABEL_KEY])

    return outputs
