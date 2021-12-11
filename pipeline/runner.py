from distutils.util import strtobool

import kfp
from argparse import Namespace
from typing import Text

from tfx.orchestration import data_types
from tfx.orchestration.kubeflow import kubeflow_dag_runner

from config import Config
from pipeline import PipelineBuilder


class Runner(object):

    @staticmethod
    def run(pipeline_args: Namespace):
        PipelineBuilder.build(pipeline_args)


def main():
    pipeline_args = Namespace()
    pipeline_args.ai_platform_training_args = {
        'project': Config.PROJECT_ID,
        'region': Config.GCP_REGION,
        'serviceAccount': Config.CUSTOM_SERVICE_ACCOUNT,
        'masterConfig': {
            'imageUri': Config.TFX_IMAGE,
        }
    }

    pipeline_args.ai_platform_serving_args = {
        'project_id': Config.PROJECT_ID,
        'model_name': Config.MODEL_NAME,
        'runtimeVersion': Config.RUNTIME_VERSION,
        'pythonVersion': Config.PYTHON_VERSION,
        'regions': [Config.GCP_REGION]
    }

    beam_tmp_folder = '{}/beam/tmp'.format(Config.ARTIFACT_STORE_URI)
    pipeline_args.beam_pipeline_args = [
        '--runner=DataflowRunner',
        '--experiments=shuffle_mode=auto',
        '--project=' + Config.PROJECT_ID,
        '--temp_location=' + beam_tmp_folder,
        '--region=' + Config.GCP_REGION,
    ]

    pipeline_args.data_root_uri = data_types.RuntimeParameter(
        name='data-root-uri',
        default=Config.DATA_ROOT_URI,
        ptype=Text
    )

    pipeline_args.train_steps = data_types.RuntimeParameter(
        name='train-steps',
        default=5000,
        ptype=int
    )

    pipeline_args.eval_steps = data_types.RuntimeParameter(
        name='eval-steps',
        default=500,
        ptype=int
    )

    pipeline_args.pipeline_root = '{}/{}/{}'.format(
        Config.ARTIFACT_STORE_URI,
        Config.PIPELINE_NAME,
        kfp.dsl.RUN_ID_PLACEHOLDER)

    pipeline_args.metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()

    pipeline_args.runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=pipeline_args.metadata_config,
        pipeline_operator_funcs=kubeflow_dag_runner.get_default_pipeline_operator_funcs(
            strtobool(Config.USE_KFP_SA)),
        tfx_image=Config.TFX_IMAGE)



if __name__ == '__main__':
    main()
