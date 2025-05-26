"""TFX pipeline runner for Kubeflow."""

# import kfp # kfp is used for kfp.dsl.RUN_ID_PLACEHOLDER, ensure it's compatible with your TFX version
from argparse import Namespace
from typing import Text

import kfp # Added import for kfp, as kfp.dsl.RUN_ID_PLACEHOLDER is used.
from tfx.orchestration import data_types
from tfx.orchestration.kubeflow import kubeflow_dag_runner

from config import Config # pylint: disable=import-error # Assuming config.py is in the same directory or PYTHONPATH
from pipeline import PipelineBuilder # pylint: disable=import-error # Assuming pipeline.py is in the same directory or PYTHONPATH


class Runner(object):
    """
    Orchestrates the TFX pipeline execution using KubeflowDagRunner.
    """

    @staticmethod
    def run(pipeline_args: Namespace):
        """
        Builds and runs the TFX pipeline.

        Args:
            pipeline_args: A Namespace object containing all arguments
                           required for pipeline construction and execution.
        """
        # Build the TFX pipeline definition
        pipeline = PipelineBuilder.build(pipeline_args=pipeline_args)

        # Configure and run the pipeline using KubeflowDagRunner
        # KubeflowDagRunner translates the TFX pipeline into a KFP pipeline
        # and submits it to the KFP engine.
        kubeflow_dag_runner.KubeflowDagRunner(
            config=pipeline_args.runner_config
        ).run(pipeline)


def main():
    """
    Main function to configure and run the TFX pipeline.
    This function sets up all the necessary arguments and configurations
    for the pipeline, then invokes the Runner.
    """
    pipeline_args = Namespace()

    # AI Platform Training arguments
    # These arguments configure the TFX Trainer component to run on Google Cloud AI Platform Training.
    # Ensure Config.TFX_IMAGE is a TFX image compatible with your target Kubeflow/KFP version
    # and AI Platform Training.
    pipeline_args.ai_platform_training_args = {
        'project': Config.PROJECT_ID,
        'region': Config.GCP_REGION,
        'serviceAccount': Config.CUSTOM_SERVICE_ACCOUNT, # Ensure this SA has necessary permissions
        'masterConfig': {
            'imageUri': Config.TFX_IMAGE, # Critical: Use a TFX image compatible with your KFP version
        }
    }

    # AI Platform Serving arguments
    # These arguments configure the TFX Pusher component to deploy models to Google Cloud AI Platform Serving.
    pipeline_args.ai_platform_serving_args = {
        'project_id': Config.PROJECT_ID,
        'model_name': Config.MODEL_NAME,
        'runtimeVersion': Config.RUNTIME_VERSION, # Ensure this runtime is supported
        'pythonVersion': Config.PYTHON_VERSION,   # Ensure this Python version is supported
        'regions': [Config.GCP_REGION]
    }

    # Beam pipeline arguments for Dataflow
    # These arguments configure TFX components that use Apache Beam to run on Dataflow.
    beam_tmp_folder = '{}/beam/tmp'.format(Config.ARTIFACT_STORE_URI)
    pipeline_args.beam_pipeline_args = [
        '--runner=DataflowRunner',
        '--experiments=shuffle_mode=auto', # Recommended for Dataflow
        '--project=' + Config.PROJECT_ID,
        '--temp_location=' + beam_tmp_folder,
        '--region=' + Config.GCP_REGION,
        # Consider adding '--sdk_container_image' if using custom containers with Dataflow Runner v2
        # and your TFX_IMAGE is suitable, or a dedicated Beam SDK container image.
    ]

    # Runtime parameter for the root directory of the input data.
    # This allows overriding the data root URI at pipeline submission time.
    pipeline_args.data_root_uri = data_types.RuntimeParameter(
        name='data-root-uri',
        default=Config.DATA_ROOT_URI, # Default value from config
        ptype=Text
    )

    # Runtime parameter for the number of training steps.
    pipeline_args.train_steps = data_types.RuntimeParameter(
        name='train-steps',
        default=5000, # Default number of training steps
        ptype=int
    )

    # Runtime parameter for the number of evaluation steps.
    pipeline_args.eval_steps = data_types.RuntimeParameter(
        name='eval-steps',
        default=500, # Default number of evaluation steps
        ptype=int
    )

    # Pipeline root for storing TFX artifacts.
    # It's crucial that this path is accessible by all Kubeflow pipeline components.
    # kfp.dsl.RUN_ID_PLACEHOLDER ensures each run has a unique artifact location.
    pipeline_args.pipeline_root = '{}/{}/{}'.format(
        Config.ARTIFACT_STORE_URI,
        Config.PIPELINE_NAME,
        kfp.dsl.RUN_ID_PLACEHOLDER
    )

    # Metadata configuration for Kubeflow Pipelines.
    # This uses the default TFX configuration for KFP metadata.
    pipeline_args.metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()

    # KubeflowDagRunner configuration
    # This configures how TFX interacts with Kubeflow Pipelines.

    # Replace strtobool from distutils.util as it's deprecated (removed in Python 3.12).
    # Assuming Config.USE_KFP_SA is a string like "true" or "false".
    # This flag is passed to `use_generic_launcher` in `get_default_pipeline_operator_funcs`.
    # Setting `use_generic_launcher=True` is generally recommended for KFP v2 compatibility,
    # as it uses a launcher image that works well with the KFP v2 backend.
    use_kfp_sa_bool = Config.USE_KFP_SA.lower() == 'true' if isinstance(Config.USE_KFP_SA, str) else bool(Config.USE_KFP_SA)


    pipeline_args.runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=pipeline_args.metadata_config,
        # get_default_pipeline_operator_funcs configures KFP pod settings.
        # The boolean argument here controls `use_generic_launcher`.
        # For newer KFP versions (especially v2 compatible), using the generic launcher is often preferred.
        pipeline_operator_funcs=kubeflow_dag_runner.get_default_pipeline_operator_funcs(
            use_kfp_sa_bool # Pass the evaluated boolean
        ),
        tfx_image=Config.TFX_IMAGE # This image is used for TFX components running on KFP.
                                   # Must be compatible with your KFP version and TFX library version.
    )

    # Execute the pipeline
    Runner.run(pipeline_args=pipeline_args)


if __name__ == '__main__':
    main()