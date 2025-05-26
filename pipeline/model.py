import functools
from typing import List, Text, Dict

import absl  # For logging
import keras_tuner  # KerasTuner for hyperparameter tuning
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs  # TrainerFnArgs is now FnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.tuner.component import TunerFnResult
from tfx_bsl.tfxio import dataset_options

# Assuming features.py is in the same directory or PYTHONPATH
from features import Feature

# Define training parameters as constants. These can be overridden by hyperparameters if tuning.
EPOCHS = 1  # Number of epochs for training. Can be tuned.
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64


def _gzip_reader_fn(filenames: List[Text]) -> tf.data.TFRecordDataset:
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _get_serve_tf_examples_fn(model: tf.keras.Model,
                              tf_transform_output: tft.TFTransformOutput
                              ) -> tf.types.experimental.ConcreteFunction:
    """
    Returns a function that parses a serialized tf.Example, applies TFT, and returns predictions.
    This function is used to create the serving signature for the exported model.
    """

    # Get the TransformFeaturesLayer from the TFTransformOutput.
    # This layer applies the transformations defined in preprocessing_fn.
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_examples: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Returns the output to be used in the serving signature."""
        # Define the feature spec for parsing raw examples.
        # Exclude the label key as it's not present during serving.
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(Feature.LABEL_KEY, None)  # Use .pop with default None

        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        # Apply the transformations.
        transformed_features = model.tft_layer(parsed_features)

        # Get predictions from the model.
        predictions = model(transformed_features)
        # For classification, it's common to return probabilities or class IDs.
        # Assuming the last layer is softmax, predictions are probabilities.
        return {'output_0': predictions}  # Adjust output key name as needed

    return serve_tf_examples_fn


def _input_fn(file_pattern: List[Text],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
    """
    Generates features and label for tuning/training from TFRecord files.

    Args:
        file_pattern: List of paths or patterns of input TFRecord files.
        data_accessor: DataAccessor for converting input to RecordBatch.
        tf_transform_output: A TFTransformOutput object from the Transform component.
        batch_size: The number of consecutive elements of returned dataset to combine in a single batch.

    Returns:
        A tf.data.Dataset that yields (features, labels) tuples.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=Feature.transformed_name(Feature.LABEL_KEY)  # Use transformed label key
        ),
        tf_transform_output.transformed_metadata.schema
    ).repeat()  # Add repeat for multiple epochs; steps_per_epoch will control epoch length


def _get_hyperparameters() -> keras_tuner.HyperParameters:
    """Returns hyperparameters for building Keras model (default search space for Tuner)."""
    hp = keras_tuner.HyperParameters()
    # Defines search space for learning rate.
    hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4], default=1e-3)
    # Defines search space for number of hidden layers.
    hp.Int('n_layers', 1, 2, default=1)
    # Conditional hyperparameters for units in each layer.
    with hp.conditional_scope('n_layers', 1):
        hp.Int('n_units_1', min_value=8, max_value=128, step=8, default=8)
    with hp.conditional_scope('n_layers', 2):
        # For n_layers == 2, n_units_1 is already defined if we don't nest scopes.
        # KerasTuner reuses hp names if defined at the same level.
        # If you want distinct n_units_1 for n_layers=1 vs n_layers=2, use different names or ensure proper scope.
        # The current setup implies n_units_1 is chosen, and if n_layers=2, n_units_2 is also chosen.
        hp.Int('n_units_1', min_value=8, max_value=128, step=8, default=8)  # This will be the same 'n_units_1' as above
        hp.Int('n_units_2', min_value=8, max_value=128, step=8, default=8)
    return hp


def _build_keras_model(hparams: keras_tuner.HyperParameters,
                       tf_transform_output: tft.TFTransformOutput) -> tf.keras.Model:
    """
    Creates a Keras Wide & Deep model.

    Args:
        hparams: A KerasTuner HyperParameters object containing tuning choices.
        tf_transform_output: A TFTransformOutput object from the Transform component.

    Returns:
        A compiled Keras model.
    """
    # Create input layers for numeric features.
    deep_feature_columns = [
        tf.feature_column.numeric_column(
            key=Feature.transformed_name(key),
            shape=()  # Scalar numeric feature
        )
        for key in Feature.NUMERIC_FEATURE_KEYS
    ]

    # Create input layers for all features based on the transformed schema.
    input_layers = {
        Feature.transformed_name(key): tf.keras.layers.Input(
            name=Feature.transformed_name(key), shape=(), dtype=tf.float32)
        for key in Feature.NUMERIC_FEATURE_KEYS
    }

    # Create categorical feature columns with identity (already integerized by TFT).
    categorical_feature_columns = [
        tf.feature_column.categorical_column_with_identity(
            key=Feature.transformed_name(key),
            # Get vocab size from TFT metadata for the transformed feature.
            num_buckets=tf_transform_output.vocabulary_size_by_name(
                Feature.transformed_name(key)  # Use transformed name if vocab was named that way
                # Or, if vocab_filename was the original key in preprocessing_fn:
                # tf_transform_output.vocabulary_size_by_name(key)
            ) if tf_transform_output.vocabulary_size_by_name(Feature.transformed_name(key)) is not None
            else tf_transform_output.num_buckets_for_transformed_feature(Feature.transformed_name(key)),
            # Fallback for older TFT
            default_value=0  # Default value for OOV
        )
        for key in Feature.CATEGORICAL_FEATURE_KEYS
    ]

    # Convert categorical identity columns to indicator columns for the wide part.
    wide_feature_columns = [
        tf.feature_column.indicator_column(categorical_column)
        for categorical_column in categorical_feature_columns
    ]

    # Add input layers for categorical features.
    input_layers.update({
        Feature.transformed_name(key): tf.keras.layers.Input(
            name=Feature.transformed_name(key), shape=(), dtype=tf.int64)  # Dtype should be int for vocab indices
        for key in Feature.CATEGORICAL_FEATURE_KEYS
    })

    # Deep part of the model.
    deep = tf.keras.layers.DenseFeatures(deep_feature_columns, name='deep_features')(input_layers)
    for i in range(int(hparams.get('n_layers'))):
        deep = tf.keras.layers.Dense(
            units=hparams.get(f'n_units_{i + 1}'),  # Use f-string for unit names
            activation='relu',
            name=f'deep_dense_{i + 1}'
        )(deep)

    # Wide part of the model.
    wide = tf.keras.layers.DenseFeatures(wide_feature_columns, name='wide_features')(input_layers)

    # Combine deep and wide parts.
    combined = tf.keras.layers.concatenate([deep, wide], name='concatenate')

    # Output layer for classification.
    output = tf.keras.layers.Dense(
        Feature.NUM_CLASSES, activation='softmax', name='output_softmax'
    )(combined)

    model = tf.keras.Model(inputs=input_layers, outputs=output)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hparams.get('learning_rate')),  # Use learning_rate
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )
    model.summary(print_fn=absl.logging.info)  # Log model summary

    return model


# TFX Tuner will call this function.
def tuner_fn(fn_args: FnArgs) -> TunerFnResult:  # TrainerFnArgs is now FnArgs
    """
    Build the tuner using the KerasTuner API.
    This function is called by the TFX Tuner component to set up and run hyperparameter tuning.

    Args:
        fn_args: Holds arguments as name/value pairs from the TFX Tuner component.
                 Key attributes include:
                 - working_dir: Directory for KerasTuner to store trial results.
                 - train_files: List of file paths for training data.
                 - eval_files: List of file paths for evaluation data.
                 - train_steps: Number of training steps per trial.
                 - eval_steps: Number of evaluation steps per trial.
                 - schema_path: Path to the schema file.
                 - transform_graph_path: Path to the transform graph.
                 - data_accessor: Utility to access data.

    Returns:
        A TunerFnResult namedtuple containing the tuner object and fitting arguments.
    """
    # Load the transform graph from the path provided by TFX.
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create a partial function for building the Keras model, fixing the tf_transform_output.
    build_keras_model_fn = functools.partial(
        _build_keras_model, tf_transform_output=tf_transform_output)

    # Initialize the KerasTuner (e.g., BayesianOptimization, Hyperband, RandomSearch).
    tuner = keras_tuner.BayesianOptimization(
        build_keras_model_fn,
        objective=keras_tuner.Objective('val_accuracy', 'max'),  # Monitor validation accuracy
        max_trials=10,  # Maximum number of hyperparameter combinations to try
        hyperparameters=_get_hyperparameters(),  # Get the search space
        # allow_new_entries and tune_new_entries are for Hyperband/RandomSearch with custom HPs.
        # Not typically used with BayesianOptimization in this way.
        directory=fn_args.working_dir,  # Directory to store tuning results
        project_name='covertype_tuning'  # Project name for KerasTuner
    )

    # Create training and evaluation datasets.
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=TRAIN_BATCH_SIZE
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=EVAL_BATCH_SIZE
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,  # Steps per epoch for training
            'validation_steps': fn_args.eval_steps,  # Steps for validation
            'epochs': EPOCHS,  # Number of epochs for each trial (can also be tuned)
            # Callbacks can be added here if needed for each trial, e.g., EarlyStopping
            # 'callbacks': [tf.keras.callbacks.EarlyStopping(patience=2)]
        }
    )


# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):  # TrainerFnArgs is now FnArgs
    """
    Train the model based on given args.
    This function is called by the TFX Trainer component.

    Args:
        fn_args: Holds arguments as name/value pairs from the TFX Trainer component.
                 Key attributes include:
                 - train_files, eval_files, train_steps, eval_steps
                 - transform_output: Path to the transform graph.
                 - serving_model_dir: Directory to save the trained model.
                 - hyperparameters: Tuned hyperparameters from the Tuner component (if used).
                 - model_run_dir: Directory for logs (e.g., TensorBoard).
                 - data_accessor: Utility to access data.
    """
    # Load the transform graph.
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Create training and evaluation datasets.
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        TRAIN_BATCH_SIZE
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        EVAL_BATCH_SIZE
    )

    # Load hyperparameters if provided by Tuner, otherwise use defaults.
    if fn_args.hyperparameters:
        hparams = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
    else:
        hparams = _get_hyperparameters()  # Use default search space (effectively default values)
    absl.logging.info('HyperParameters for training: %s', hparams.get_config())

    # Build the Keras model within a distribution strategy scope for potential distributed training.
    # MirroredStrategy is for single-machine, multi-GPU.
    # For multi-worker distributed training (e.g., on AI Platform),
    # TFX/TensorFlow will set up the appropriate strategy.
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = _build_keras_model(
            hparams=hparams,
            tf_transform_output=tf_transform_output
        )

    # TensorBoard callback for logging training progress.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch'
    )

    # Train the model.
    model.fit(
        train_dataset,
        epochs=EPOCHS,  # Use the defined number of epochs
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback]
    )

    # Define serving signatures.
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model, tf_transform_output
        ),
        # You can add more signatures here if needed.
    }

    # Save the model with signatures.
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
