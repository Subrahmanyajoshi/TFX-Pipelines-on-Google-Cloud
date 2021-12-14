from argparse import Namespace
from distutils.util import strtobool

from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform
from tfx.proto import example_gen_pb2

from pipeline.config import Config


class PipelineBuilder(object):

    TRANSFORM_MODULE = 'preprocessing.py'

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
