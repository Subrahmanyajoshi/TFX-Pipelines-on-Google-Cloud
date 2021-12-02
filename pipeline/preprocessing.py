import os
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import TextVectorization

from features import Feature

FEATURE_KEY = 'input'
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

MAX_SEQUENCE_LENGTH = 250
VOCAB_SIZE = 20000

int_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)


def preprocessing_fn(inputs, custom_config):
    """Preprocesses Dataset."""
    
    tokenizer_path = custom_config.get('tokenizer_path')
    
    os.system(f'gsutil -m cp {tokenizer_path} ./')
    
    with open(os.path.basename(tokenizer_path), 'rb') as handle:
        vectorizer_layer = pickle.load(handle)

    outputs = {}
    
    key = Feature.transformed_name(Feature.FEATURE_KEY)
    # Fill in missing values and create dense tensors.    
    
    
    # int_vectorize_layer.adapt(inputs[Feature.FEATURE_KEY])
    
    text = tf.expand_dims(inputs[Feature.FEATURE_KEY], -1)
    outputs[key] = int_vectorize_layer(text)
    
    
    # outputs[key] = _fill_in_missing(inputs[Feature.FEATURE_KEY])
    # outputs[key] = tokenizer.texts_to_sequences(outputs[key])
    # outputs[key] = sequence.pad_sequences(outputs[key], maxlen=MAX_SEQUENCE_LENGTH)

    # Convert label to dense tensor.
    outputs[Feature.transformed_name(Feature.LABEL_KEY)] = _fill_in_missing(inputs[Feature.LABEL_KEY])

    return outputs
