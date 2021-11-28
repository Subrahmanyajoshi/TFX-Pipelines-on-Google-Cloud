from typing import List, Text

import absl
import keras_tuner
import tensorflow as tf
import tensorflow_transform as tft
from features import Feature

from tfx.components.trainer.executor import TrainerFnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.tuner.component import TunerFnResult
from tfx_bsl.tfxio import dataset_options


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""
    model.tft_layer = tf_transform_output.transform_Feature_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(Feature.LABEL_KEY)
        parsed_feature = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_feature = model.tft_layer(parsed_feature)

        return model(transformed_feature)

    return serve_tf_examples_fn


def _input_fn(file_pattern: List[Text],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
    """Generates Feature and label for tuning/training.
    Args:
      file_pattern: List of paths or patterns of input tfrecord files.
      data_accessor: DataAccessor for converting input to RecordBatch.
      tf_transform_output: A TFTransformOutput.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch
    Returns:
      A dataset that contains (Feature, indices) tuple where Feature is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=Feature.transformed_name(Feature.LABEL_KEY)),
        tf_transform_output.transformed_metadata.schema)

    return dataset


def _get_hyperparameters() -> keras_tuner.HyperParameters:
    """Returns hyperparameters for building Keras model."""
    hp = keras_tuner.HyperParameters()

    # Defines search space.
    hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4], default=1e-3)
    hp.Int('n_layers', 1, 2, default=1)

    with hp.conditional_scope('n_layers', 1):
        hp.Int('n_units_1', min_value=8, max_value=128, step=8, default=8)

    with hp.conditional_scope('n_layers', 2):
        hp.Int('n_units_1', min_value=8, max_value=128, step=8, default=8)
        hp.Int('n_units_2', min_value=8, max_value=128, step=8, default=8)

    return hp


def _build_keras_model(hparams: keras_tuner.HyperParameters,
                       tf_transform_output: tft.TFTransformOutput) -> tf.keras.Model:
    """Creates a Keras WideDeep Classifier model.
    Args:
      hparams: Holds HyperParameters for tuning.
      tf_transform_output: A TFTransformOutput.
    Returns:
      A keras Model.
    """
    deep_columns = [
        tf.feature_column.numeric_column(
            key=Feature.transformed_name(key),
            shape=())
        for key in Feature.NUMERIC_FEATURE_KEYS
    ]

    input_layers = {
        column.key: tf.keras.layers.Input(name=column.key, shape=(), dtype=tf.float32)
        for column in deep_columns
    }

    categorical_columns = [
        tf.feature_column.categorical_column_with_identity(
            key=Feature.transformed_name(key),
            num_buckets=tf_transform_output.num_buckets_for_transformed_feature(Feature.transformed_name(key)),
            default_value=0)
        for key in Feature.CATEGORICAL_FEATURE_KEYS
    ]

    wide_columns = [
        tf.feature_column.indicator_column(categorical_column)
        for categorical_column in categorical_columns
    ]

    input_layers.update({
        column.categorical_column.key: tf.keras.layers.Input(name=column.categorical_column.key, shape=(),
                                                             dtype=tf.int32)
        for column in wide_columns
    })

    deep = tf.keras.layers.DenseFeature(deep_columns)(input_layers)
    for n in range(int(hparams.get('n_layers'))):
        deep = tf.keras.layers.Dense(units=hparams.get('n_units_' + str(n + 1)))(deep)

    wide = tf.keras.layers.DenseFeature(wide_columns)(input_layers)

    output = tf.keras.layers.Dense(Feature.NUM_CLASSES, activation='softmax')(
        tf.keras.layers.concatenate([deep, wide]))

    model = tf.keras.Model(input_layers, output)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=hparams.get('learning_rate')),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.summary(print_fn=absl.logging.info)

    return model
