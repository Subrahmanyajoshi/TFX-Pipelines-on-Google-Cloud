import tensorflow as tf
import tensorflow_transform as tft


class Feature(object):
    FEATURE_KEYS = ['input']
    LABEL_KEY = 'labels'
    
    def transformed_name(key):
        return key + '_xf'

    
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
    
    # Fill in missing values and create dense tensors.
    for key in Feature.FEATURE_KEYS:
        outputs[Feature.transformed_name(key)] = _fill_in_missing(inputs[key])

    # Convert label to dense tensor.
    outputs[Feature.transformed_name(Feature.LABEL_KEY)] = _fill_in_missing(inputs[Feature.LABEL_KEY])

    return outputs

