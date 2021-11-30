import functools
from typing import List, Text

import absl
import keras_tuner
import tensorflow as tf
import tensorflow_transform as tft
from features import Feature
from tensorflow.keras import layers

from tfx.components.trainer.executor import TrainerFnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.tuner.component import TunerFnResult
from tfx_bsl.tfxio import dataset_options

EPOCHS = 1
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64


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
        for key in Feature.FEATURE_KEYS
    ]

    input_layers = {
        column.key: tf.keras.layers.Input(name=column.key, shape=(), dtype=tf.float32)
        for column in deep_columns
    }

    deep = layers.DenseFeatures(deep_columns)(input_layers)
    for n in range(int(hparams.get('n_layers'))):
        deep = layers.Dense(units=hparams.get('n_units_' + str(n + 1)))(deep)

    output = layers.Dense(Feature.NUM_CLASSES, activation='softmax')(deep)

    model = tf.keras.Model(input_layers, output)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=hparams.get('learning_rate')),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.summary(print_fn=absl.logging.info)

    return model


# TFX Tuner will call this function.
def tuner_fn(fn_args: TrainerFnArgs) -> TunerFnResult:
    """Build the tuner using the KerasTuner API.
    Args:
      fn_args: Holds args as name/value pairs.
        - working_dir: working dir for tuning.
        - train_files: List of file paths containing training tf.Example data.
        - eval_files: List of file paths containing eval tf.Example data.
        - train_steps: number of train steps.
        - eval_steps: number of eval steps.
        - schema_path: optional schema of the input data.
        - transform_graph_path: optional transform graph produced by TFT.
    Returns:
      A namedtuple contains the following:
        - tuner: A BaseTuner that will be used for tuning.
        - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                      model , e.g., the training and validation dataset. Required
                      args depend on the above tuner's implementation.
    """
    transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Construct a build_keras_model_fn that just takes hyperparams from get_hyperparameters as input.
    build_keras_model_fn = functools.partial(
        _build_keras_model, tf_transform_output=transform_graph)

    # BayesianOptimization is a subclass of kerastuner.Tuner which inherits from BaseTuner.
    tuner = keras_tuner.BayesianOptimization(
        build_keras_model_fn,
        max_trials=10,
        hyperparameters=_get_hyperparameters(),
        # New entries allowed for n_units hyperparameter construction conditional on n_layers selected.
        #       allow_new_entries=True,
        #       tune_new_entries=True,
        objective=keras_tuner.Objective('val_sparse_categorical_accuracy', 'max'),
        directory=fn_args.working_dir,
        project_name='covertype_tuning')

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        transform_graph,
        batch_size=TRAIN_BATCH_SIZE)

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        transform_graph,
        batch_size=EVAL_BATCH_SIZE)

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        })


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
    """Train the model based on given args.
    Args:
      fn_args: Holds args used to train the model as name/value pairs.
    """

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        TRAIN_BATCH_SIZE)

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        EVAL_BATCH_SIZE)

    if fn_args.hyperparameters:
        hparams = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
    else:
        # This is a shown case when hyperparameters is decided and Tuner is removed
        # from the pipeline. User can also inline the hyperparameters directly in
        # _build_keras_model.
        hparams = _get_hyperparameters()
    absl.logging.info('HyperParameters for training: %s' % hparams.get_config())

    # Distribute training over multiple replicas on the same machine.
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = _build_keras_model(
            hparams=hparams,
            tf_transform_output=tf_transform_output)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch')

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback])

    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(model,
                                      tf_transform_output).get_concrete_function(
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name='examples')),
    }

    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)