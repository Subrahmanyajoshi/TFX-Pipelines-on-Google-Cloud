import os

class Config:
    """
    Sets configuration variables for the TFX pipeline.
    These configurations are typically sourced from environment variables,
    allowing for flexible deployment across different environments.
    """
    # Lab user environment resource settings
    # Google Cloud Project ID where the pipeline and resources will be deployed.
    PROJECT_ID = os.getenv("PROJECT_ID", "text-analysis-323506")
    # Google Cloud region for deploying resources like AI Platform jobs and KFP.
    GCP_REGION = os.getenv("GCP_REGION", "us-central1")
    # Google Cloud Storage URI for storing TFX pipeline artifacts.
    # Ensure this bucket exists and the KFP/AI Platform service accounts have access.
    ARTIFACT_STORE_URI = os.getenv("ARTIFACT_STORE_URI", "gs://text-analysis-323506-artifact-store")
    # Custom service account for AI Platform Training/Serving and other GCP services.
    # This SA needs appropriate permissions (e.g., GCS access, AI Platform access).
    CUSTOM_SERVICE_ACCOUNT = os.getenv("CUSTOM_SERVICE_ACCOUNT",
                                       "my-api-sa@text-analysis-323506.iam.gserviceaccount.com")

    # Lab user runtime environment settings
    # Name of the TFX pipeline.
    PIPELINE_NAME = os.getenv("PIPELINE_NAME", "covertype_training")
    # Name of the model to be deployed on AI Platform Serving.
    MODEL_NAME = os.getenv("MODEL_NAME", "covertype_classifier")
    # Root URI for the input data used by CsvExampleGen.
    DATA_ROOT_URI = os.getenv("DATA_ROOT_URI", "gs://text-analysis-323506/covertype")

    # CRITICAL: TFX Docker image for Kubeflow Pipelines components and AI Platform Training.
    # This image MUST be compatible with your TFX library version and your target Kubeflow Pipelines version.
    # For "latest Kubeflow", use a recent TFX image.
    # Examples:
    # - Specific version: "gcr.io/tfx-oss-public/tfx:1.15.0" (replace 1.15.0 with the desired TFX version)
    # - Latest (use with caution, prefer specific versions for production): "tensorflow/tfx:latest"
    # Ensure the TFX version in this image aligns with the TFX library version used to define and run the pipeline.
    TFX_IMAGE = os.getenv("KUBEFLOW_TFX_IMAGE", "gcr.io/tfx-oss-public/tfx:1.15.0") # Example: Updated to a more recent TFX version

    # AI Platform Serving runtime and Python versions.
    # These should be compatible with the TensorFlow version used to train your model
    # (which is determined by the TFX_IMAGE).
    # Check Google Cloud AI Platform Serving documentation for supported runtimes.
    # For TF 2.x models, common runtimes are "2.x".
    RUNTIME_VERSION = os.getenv("RUNTIME_VERSION", "2.15") # Example: Updated to match a potential TF version in TFX 1.15.0
    PYTHON_VERSION = os.getenv("PYTHON_VERSION", "3.10")   # Example: Updated to match a potential Python version

    # Flag to control whether Kubeflow Pipelines service account is used for TFX components.
    # Stored as a string ("True" or "False") and converted to boolean in the pipeline script.
    # For newer KFP versions, using the KFP SA (via use_generic_launcher=True) is often recommended.
    USE_KFP_SA = os.getenv("USE_KFP_SA", "True") # Recommended to be "True" for modern KFP

    # Flag to enable/disable the Tuner component in the pipeline.
    # Stored as a string ("True" or "False") and converted to boolean in the pipeline script.
    ENABLE_TUNING = os.getenv("ENABLE_TUNING", "True")
