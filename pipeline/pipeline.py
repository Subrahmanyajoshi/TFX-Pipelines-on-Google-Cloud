from argparse import Namespace
from distutils.util import strtobool

from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Tuner
from tfx.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.components.common.importer import Importer
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.proto import example_gen_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import HyperParameters, Model, ModelBlessing

from pipeline.config import Config


class PipelineBuilder(object):
    TRANSFORM_MODULE = 'preprocessing.py'
    TRAINER_MODULE_FILE = 'model.py'

    @staticmethod
    def build(pipeline_args: Namespace):
        pipeline_name = Config.PIPELINE_NAME,
        pipeline_root = pipeline_args.pipeline_root,
        data_root_uri = pipeline_args.data_root_uri,
        train_steps = pipeline_args.train_steps,
        eval_steps = pipeline_args.eval_steps,
        enable_tuning = strtobool(Config.ENABLE_TUNING),
        ai_platform_training_args = pipeline_args.ai_platform_training_args,
        ai_platform_serving_args = pipeline_args.ai_platform_serving_args,
        beam_pipeline_args = pipeline_args.beam_pipeline_args

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

        tuner = Tuner(
            module_file=PipelineBuilder.TRAINER_MODULE_FILE,
            examples=transform.outputs['transformed_examples'],
            transform_graph=transform.outputs['transform_graph'],
            train_args=trainer_pb2.TrainArgs(num_steps=1000),
            eval_args=trainer_pb2.EvalArgs(num_steps=500))

        hparams_importer = Importer(
            source_uri=tuner.outputs['best_hyperparameters'].get()[0].uri,
            artifact_type=HyperParameters).with_id('hparams_importer')

        trainer = Trainer(
            custom_executor_spec=executor_spec.ExecutorClassSpec(trainer_executor.GenericExecutor),
            module_file=PipelineBuilder.TRAINER_MODULE_FILE,
            transformed_examples=transform.outputs['transformed_examples'],
            schema=schema_gen.outputs['schema'],
            transform_graph=transform.outputs['transform_graph'],
            hyperparameters=hparams_importer.outputs['result'],
            train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=5000),
            eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=1000))

        model_resolver = Resolver(
            strategy_class=LatestBlessedModelStrategy,
            model=Channel(type=Model),
            model_blessing=Channel(type=ModelBlessing)
        ).with_id('latest_blessed_model_resolver')
