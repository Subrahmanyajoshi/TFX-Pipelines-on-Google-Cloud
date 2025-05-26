import tensorflow as tf
import tensorflow_transform as tft


# Assuming features.py is in the same directory or PYTHONPATH
# from features import Feature # This should be uncommented if Feature class is used directly here.
# For this example, we'll assume Feature is passed or its constants are known.

# It's good practice to define constants for feature names,
# e.g., by importing them from a 'features.py' module.
# For demonstration, if 'features.py' defines:
# class Feature:
#     NUMERIC_FEATURE_KEYS = [...]
#     CATEGORICAL_FEATURE_KEYS = [...]
#     LABEL_KEY = 'Cover_Type'
#     @staticmethod
#     def transformed_name(key): return key + '_xf'

def _fill_in_missing(x: tf.SparseTensor) -> tf.Tensor:
    """Replace missing values in a SparseTensor.
    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.

    Args:
        x: A `SparseTensor` of rank 2. Its dense shape should have size at most 1
           in the second dimension (i.e., a single value per example).

    Returns:
        A rank 1 dense tensor where missing values of `x` have been filled in.
    """
    # Ensure x is a SparseTensor, common for CSV inputs where columns might be missing.
    if not isinstance(x, tf.SparseTensor):
        # If x is already dense, it might mean it's not truly "missing" in the sparse sense,
        # or it's a feature that's always present.
        # Depending on the use case, you might return x directly or handle it differently.
        # For this function's original intent, we assume it's for sparse inputs.
        # If you expect dense inputs that might contain specific "missing" markers (e.g., NaN for numeric),
        # that would require different handling.
        # For now, let's assume it's sparse or can be treated as such for densification.
        # This part might need adjustment based on actual data characteristics.
        # A simple pass-through if already dense and not string (for string, empty is fine).
        if x.dtype != tf.string:
            return tf.squeeze(x, axis=1)  # Squeeze if it has an extra dimension

    default_value = '' if x.dtype == tf.string else tf.cast(0, x.dtype)  # Match dtype for numeric

    # tf.sparse.to_dense requires a default value.
    # The input SparseTensor x is expected to be of shape [batch_size, 1] or [batch_size, 0] effectively.
    # We ensure it's treated as [batch_size, 1] for to_dense.
    dense_tensor = tf.sparse.to_dense(
        tf.sparse.reset_shape(x, new_shape=[x.dense_shape[0], 1]),  # Ensure last dim is 1
        default_value
    )
    return tf.squeeze(dense_tensor, axis=1)


def preprocessing_fn(inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """
    Preprocesses raw input features using TensorFlow Transform.

    This function is applied by the TFX Transform component. It defines
    the feature engineering logic, such as scaling, vocabulary generation, etc.
    The transformations defined here are applied during training and are also
    embedded into the serving graph for consistent preprocessing at inference time.

    Args:
        inputs: A dictionary mapping feature keys to raw `Tensor` or `SparseTensor`s.

    Returns:
        A dictionary mapping transformed feature keys to `Tensor`s.
    """
    # To use Feature class constants, ensure it's imported:
    from features import Feature  # Assuming features.py is accessible

    outputs = {}

    # Scale numerical features to z-score.
    for key in Feature.NUMERIC_FEATURE_KEYS:
        # Fill missing values first, then scale.
        filled_numeric_feature = _fill_in_missing(inputs[key])
        outputs[Feature.transformed_name(key)] = tft.scale_to_z_score(
            filled_numeric_feature, name=f'scale_{key}')  # Add name for op clarity

    # Generate vocabularies and map categorical features to integers.
    for key in Feature.CATEGORICAL_FEATURE_KEYS:
        # Fill missing values first, then compute vocabulary.
        filled_categorical_feature = _fill_in_missing(inputs[key])
        outputs[Feature.transformed_name(key)] = tft.compute_and_apply_vocabulary(
            x=filled_categorical_feature,
            num_oov_buckets=1,  # Number of out-of-vocabulary buckets.
            vocab_filename=key,  # Filename for the generated vocabulary.
            name=f'vocab_{key}'  # Add name for op clarity
        )

    # Convert Label_Key to a dense tensor if it's sparse.
    # No transformation other than filling missing, as it's the label.
    outputs[Feature.transformed_name(Feature.LABEL_KEY)] = _fill_in_missing(
        inputs[Feature.LABEL_KEY])

    return outputs
