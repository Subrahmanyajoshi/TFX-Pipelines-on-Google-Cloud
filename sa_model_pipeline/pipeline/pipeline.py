from argparse import Namespace
from distutils.util import strtobool

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Tuner, \
    Evaluator, InfraValidator
from tfx.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.extensions.google_cloud_ai_platform.pusher.component import Pusher
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2, trainer_pb2, tuner_pb2, infra_validator_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

from config import Config


class PipelineBuilder(object):
    TRANSFORM_MODULE = 'preprocessing.py'
    TRAINER_MODULE_FILE = 'model.py'

    @staticmethod
    def build(pipeline_args: Namespace):
        pipeline_name = Config.PIPELINE_NAME
        pipeline_root = pipeline_args.pipeline_root
        data_root_uri = pipeline_args.data_root_uri
        train_steps = pipeline_args.train_steps
        eval_steps = pipeline_args.eval_steps
        enable_tuning = strtobool(Config.ENABLE_TUNING)
        ai_platform_training_args = pipeline_args.ai_platform_training_args
        ai_platform_serving_args = pipeline_args.ai_platform_serving_args
        beam_pipeline_args = pipeline_args.beam_pipeline_args
        enable_cache = False

        output_config = example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=4),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
            ]))

        example_gen = CsvExampleGen(
            input_base=data_root_uri,
            output_config=output_config)

        statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

        schema_gen = SchemaGen(
            statistics=statistics_gen.outputs['statistics'],
            infer_feature_shape=False)

        example_validator = ExampleValidator(
            statistics=statistics_gen.outputs['statistics'],
            schema=schema_gen.outputs['schema']).with_id(
            'example_validator')

        transform = Transform(
            examples=example_gen.outputs['examples'],
            schema=schema_gen.outputs['schema'],
            module_file=PipelineBuilder.TRANSFORM_MODULE)

        if enable_tuning:
            tuner = Tuner(
                module_file=PipelineBuilder.TRAINER_MODULE_FILE,
                examples=transform.outputs['transformed_examples'],
                transform_graph=transform.outputs['transform_graph'],
                train_args=trainer_pb2.TrainArgs(num_steps=1000),
                eval_args=trainer_pb2.EvalArgs(num_steps=500),
                tune_args=tuner_pb2.TuneArgs(
                    # num_parallel_trials=3 means that 3 search loops are running in parallel.
                    num_parallel_trials=3),
                custom_config={
                    # Configures Cloud AI Platform-specific configs. For details, see
                    # https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#traininginput.
                    ai_platform_trainer_executor.TRAINING_ARGS_KEY: ai_platform_training_args
                })

        trainer = Trainer(
            custom_executor_spec=executor_spec.ExecutorClassSpec(trainer_executor.GenericExecutor),
            module_file=PipelineBuilder.TRAINER_MODULE_FILE,
            transformed_examples=transform.outputs['transformed_examples'],
            schema=schema_gen.outputs['schema'],
            transform_graph=transform.outputs['transform_graph'],
            hyperparameters=tuner.outputs['best_hyperparameters'] if enable_tuning else None,
            train_args={'num_steps': train_steps},
            eval_args={'num_steps': eval_steps},
            custom_config={'ai_platform_training_args': ai_platform_training_args})

        model_resolver = Resolver(
            strategy_class=LatestBlessedModelStrategy,
            model=Channel(type=Model),
            model_blessing=Channel(type=ModelBlessing)).with_id('latest_blessed_model_resolver')

        accuracy_threshold = tfma.MetricThreshold(
            value_threshold=tfma.GenericValueThreshold(
                lower_bound={'value': 0.5},
                upper_bound={'value': 0.99})
        )

        metrics_specs = tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='SparseCategoricalAccuracy',
                                  threshold=accuracy_threshold),
                tfma.MetricConfig(class_name='ExampleCount')])

        eval_config = tfma.EvalConfig(
            model_specs=[
                tfma.ModelSpec(label_key='Cover_Type')],
            metrics_specs=[metrics_specs],
            slicing_specs=[
                tfma.SlicingSpec(),
                tfma.SlicingSpec(feature_keys=['Wilderness_Area'])
            ]
        )

        model_analyzer = Evaluator(
            examples=example_gen.outputs['examples'],
            model=trainer.outputs['model'],
            baseline_model=model_resolver.outputs['model'],
            eval_config=eval_config
        )

        serving_config = infra_validator_pb2.ServingSpec(
            tensorflow_serving=infra_validator_pb2.TensorFlowServing(
                tags=['latest']),
            kubernetes=infra_validator_pb2.KubernetesConfig(),
        )

        validation_config = infra_validator_pb2.ValidationSpec(
            max_loading_time_seconds=60,
            num_tries=3,
        )

        request_config = infra_validator_pb2.RequestSpec(
            tensorflow_serving=infra_validator_pb2.TensorFlowServingRequestSpec(),
            num_examples=3,
        )

        infra_validator = InfraValidator(
            model=trainer.outputs['model'],
            examples=example_gen.outputs['examples'],
            serving_spec=serving_config,
            validation_spec=validation_config,
            request_spec=request_config,
        )

        pusher = Pusher(
            model=trainer.outputs['model'],
            model_blessing=model_analyzer.outputs['blessing'],
            infra_blessing=infra_validator.outputs['blessing'],
            custom_config={tfx.extensions.google_cloud_ai_platform.experimental.PUSHER_SERVING_ARGS_KEY: \
                               ai_platform_serving_args})

        components = [
            example_gen,
            statistics_gen,
            schema_gen,
            schema_gen,
            example_validator,
            transform,
            trainer,
            model_resolver,
            model_analyzer,
            infra_validator,
            pusher
        ]

        if enable_tuning:
            components.append(tuner)

        return pipeline.Pipeline(
            pipeline_name=pipeline_name,
            pipeline_root=pipeline_root,
            components=components,
            enable_cache=enable_cache,
            beam_pipeline_args=beam_pipeline_args
        )
