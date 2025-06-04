from argparse import Namespace
# Removed: from distutils.util import strtobool # strtobool is deprecated

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx # Using tfx.v1 for stable APIs
from tfx.components import (CsvExampleGen, StatisticsGen, SchemaGen,
                            ExampleValidator, Transform, Trainer, Tuner,
                            Evaluator, InfraValidator)
from tfx.components.base import executor_spec
# trainer_executor is used for local execution if not using AI Platform
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
# For Pusher, using the non-experimental key for serving args with TFX >= 1.0
from tfx.extensions.google_cloud_ai_platform.pusher.component import Pusher
# For Trainer and Tuner custom_config with AI Platform
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2, trainer_pb2, tuner_pb2, infra_validator_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

from config import Config # Assuming config.py is in the same directory or PYTHONPATH

class PipelineBuilder(object):
    """
    Builds a TFX pipeline definition.
    This class encapsulates the logic for assembling TFX components into a coherent pipeline.
    """
    # Relative path to the preprocessing module file from the pipeline's execution context.
    TRANSFORM_MODULE = 'preprocessing.py'
    # Relative path to the model training/tuning module file.
    TRAINER_MODULE_FILE = 'model.py'

    @staticmethod
    def build(pipeline_args: Namespace):
        """
        Constructs the TFX pipeline with all its components.

        Args:
            pipeline_args: A Namespace object containing runtime parameters and
                           configurations for the pipeline (e.g., pipeline_root,
                           AI Platform args, Beam args).

        Returns:
            A tfx.orchestration.pipeline.Pipeline object.
        """
        pipeline_name = Config.PIPELINE_NAME
        pipeline_root = pipeline_args.pipeline_root
        data_root_uri = pipeline_args.data_root_uri
        train_steps = pipeline_args.train_steps.default
        eval_steps = pipeline_args.eval_steps.default


        # Convert ENABLE_TUNING from string (from Config) to boolean
        enable_tuning_str = Config.ENABLE_TUNING
        enable_tuning = enable_tuning_str.lower() == 'true' if isinstance(enable_tuning_str, str) else bool(enable_tuning_str)

        ai_platform_training_args = pipeline_args.ai_platform_training_args
        ai_platform_serving_args = pipeline_args.ai_platform_serving_args
        beam_pipeline_args = pipeline_args.beam_pipeline_args
        # Caching can be enabled for faster iterative development if inputs/configs don't change.
        enable_cache = False # Set to True to enable caching if desired

        # Configuration for CsvExampleGen to split data into 'train' and 'eval' sets.
        output_config = example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=4),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
            ]))

        # Ingests data from CSV files.
        example_gen = CsvExampleGen(
            input_base=data_root_uri,
            output_config=output_config)

        # Computes statistics over the data.
        statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

        # Infers a schema from the statistics.
        schema_gen = SchemaGen(
            statistics=statistics_gen.outputs['statistics'],
            infer_feature_shape=False) # infer_feature_shape=True can be useful for some cases

        # Validates new data against the schema.
        example_validator = ExampleValidator(
            statistics=statistics_gen.outputs['statistics'],
            schema=schema_gen.outputs['schema']).with_id(
            'example_validator') # Explicit ID for clarity

        # Performs feature engineering.
        transform = Transform(
            examples=example_gen.outputs['examples'],
            schema=schema_gen.outputs['schema'],
            module_file=PipelineBuilder.TRANSFORM_MODULE)

        tuner_component = None # Initialize tuner_component
        if enable_tuning:
            # Tunes hyperparameters for the model.
            # Uses AI Platform Training for distributed tuning trials.
            tuner_component = Tuner(
                module_file=PipelineBuilder.TRAINER_MODULE_FILE,
                examples=transform.outputs['transformed_examples'],
                transform_graph=transform.outputs['transform_graph'],
                train_args=trainer_pb2.TrainArgs(num_steps=1000), # Steps per trial for training
                eval_args=trainer_pb2.EvalArgs(num_steps=500),   # Steps per trial for evaluation
                tune_args=tuner_pb2.TuneArgs(
                    num_parallel_trials=3 # Number of tuning trials to run in parallel
                ),
                custom_config={
                    # Configures Cloud AI Platform-specific settings for each tuning trial.
                    ai_platform_trainer_executor.TRAINING_ARGS_KEY: ai_platform_training_args
                })

        # Trains the model.
        # Can use AI Platform Training for distributed training.
        trainer = Trainer(
            module_file=PipelineBuilder.TRAINER_MODULE_FILE,
            examples=transform.outputs['transformed_examples'],
            schema=schema_gen.outputs['schema'],
            transform_graph=transform.outputs['transform_graph'],
            # Uses hyperparameters from Tuner if enabled, otherwise None (model.py should handle default hparams).
            hyperparameters=tuner_component.outputs['best_hyperparameters'] if enable_tuning and tuner_component else None,
            train_args=trainer_pb2.TrainArgs(num_steps=train_steps), # Using proto for consistency
            eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps),   # Using proto for consistency
            custom_config={
                # Configures Cloud AI Platform-specific settings for the Trainer.
                # This key (TRAINING_ARGS_KEY) tells TFX to use the AI Platform custom executor.
                ai_platform_trainer_executor.TRAINING_ARGS_KEY: ai_platform_training_args
            }
        )

        # Resolver to get the latest blessed model for model validation.
        model_resolver = Resolver(
            strategy_class=LatestBlessedModelStrategy,
            model=Channel(type=Model),
            model_blessing=Channel(type=ModelBlessing)
        ).with_id('latest_blessed_model_resolver') # Explicit ID

        # Configuration for model evaluation.
        # Defines metrics, thresholds, and slicing for analysis.
        accuracy_threshold = tfma.MetricThreshold(
            value_threshold=tfma.GenericValueThreshold(
                lower_bound={'value': 0.5}, # Example: model accuracy must be > 0.5
                upper_bound={'value': 0.99}) # Example: model accuracy must be < 0.99 (sanity check)
        )

        metrics_specs = tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='SparseCategoricalAccuracy',
                                  threshold=accuracy_threshold),
                tfma.MetricConfig(class_name='ExampleCount')
            ]
        )

        eval_config = tfma.EvalConfig(
            model_specs=[
                # Specifies the label key for the model.
                tfma.ModelSpec(label_key=Config.LABEL_KEY) # Using Config.LABEL_KEY for consistency
            ],
            metrics_specs=[metrics_specs],
            slicing_specs=[
                tfma.SlicingSpec(), # Overall slice
                tfma.SlicingSpec(feature_keys=['Wilderness_Area']) # Slice by Wilderness_Area
            ]
        )

        # Evaluates the trained model against a baseline (e.g., previously blessed model).
        model_analyzer = Evaluator(
            examples=example_gen.outputs['examples'],
            model=trainer.outputs['model'],
            baseline_model=model_resolver.outputs['model'], # Optional: for model vs model evaluation
            eval_config=eval_config
        ).with_id("model_evaluator") # Explicit ID

        # Configuration for InfraValidator.
        # Specifies serving environment (TensorFlow Serving on Kubernetes) and validation parameters.
        serving_config = infra_validator_pb2.ServingSpec(
            tensorflow_serving=infra_validator_pb2.TensorFlowServing(
                tags=['latest'] # Tag for the model in TF Serving
            ),
            kubernetes=infra_validator_pb2.KubernetesConfig(), # For K8s deployment validation
        )

        validation_config = infra_validator_pb2.ValidationSpec(
            max_loading_time_seconds=60, # Max time to wait for model to load
            num_tries=3, # Number of retries for validation
        )

        request_config = infra_validator_pb2.RequestSpec(
            tensorflow_serving=infra_validator_pb2.TensorFlowServingRequestSpec(),
            num_examples=3, # Number of examples to send for validation
        )

        # Validates that the model can be loaded and served in the target infrastructure.
        infra_validator = InfraValidator(
            model=trainer.outputs['model'],
            examples=example_gen.outputs['examples'], # Sample examples for validation
            serving_spec=serving_config,
            validation_spec=validation_config,
            request_spec=request_config,
        ).with_id("infra_validator") # Explicit ID

        # Pushes the model to a deployment target (e.g., AI Platform Serving)
        # if it passes evaluation and infra validation.
        pusher = Pusher(
            model=trainer.outputs['model'],
            model_blessing=model_analyzer.outputs['blessing'],
            infra_blessing=infra_validator.outputs['blessing'], # Use blessing from InfraValidator
            custom_config={
                # Use the non-experimental key for TFX >= 1.0 for AI Platform Pusher.
                tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY: ai_platform_serving_args
            }
        ).with_id("model_pusher") # Explicit ID

        # Define the list of components for the pipeline.
        components = [
            example_gen,
            statistics_gen,
            schema_gen, # Corrected: Removed duplicate schema_gen
            example_validator,
            transform,
            # Tuner is added conditionally
            trainer,
            model_resolver,
            model_analyzer,
            infra_validator,
            pusher
        ]

        if enable_tuning and tuner_component:
            components.append(tuner_component)
            # Ensure Trainer uses the output of Tuner if tuning is enabled.
            # This is handled by the `hyperparameters` argument in the Trainer definition.

        # Construct and return the TFX pipeline.
        return pipeline.Pipeline(
            pipeline_name=pipeline_name,
            pipeline_root=pipeline_root,
            components=components,
            enable_cache=enable_cache,
            beam_pipeline_args=beam_pipeline_args,
            # metadata_connection_config can be specified here if not using default KFP metadata.
        )
