import os

class Config:
    """Sets configuration vars."""
    # Lab user environment resource settings
    GCP_REGION = os.getenv("GCP_REGION", "us-central1")
    PROJECT_ID = os.getenv("PROJECT_ID", "text-analysis-323506")
    ARTIFACT_STORE_URI = os.getenv("ARTIFACT_STORE_URI", "gs://text-analysis-323506-artifact-store")
    CUSTOM_SERVICE_ACCOUNT = os.getenv("CUSTOM_SERVICE_ACCOUNT",
                                       "my-api-sa@text-analysis-323506.iam.gserviceaccount.com")
    # Lab user runtime environment settings
    PIPELINE_NAME = os.getenv("PIPELINE_NAME", "covertype_training")
    MODEL_NAME = os.getenv("MODEL_NAME", "covertype_classifier")
    DATA_ROOT_URI = os.getenv("DATA_ROOT_URI", "gs://text-analysis-323506/covertype")
    TFX_IMAGE = os.getenv("KUBEFLOW_TFX_IMAGE", "tensorflow/tfx:1.4.0")
    RUNTIME_VERSION = os.getenv("RUNTIME_VERSION", "2.6")
    PYTHON_VERSION = os.getenv("PYTHON_VERSION", "3.7")
    USE_KFP_SA = os.getenv("USE_KFP_SA", "False")
    ENABLE_TUNING = os.getenv("ENABLE_TUNING", "True")
