"""Kubeflow Pipeline configuration for Google Cloud deployment."""

import os
from typing import Dict, List

PIPELINE_NAME = os.getenv("KF_PIPELINE_NAME", "covertype-training-pipeline")

# Kubernetes configurations
K8S_NAMESPACE = "kubeflow"
PIPELINE_ROOT = os.getenv("ARTIFACT_STORE_URI", "gs://text-analysis-323506-artifact-store")

# GCP configurations
PROJECT_ID = os.getenv("PROJECT_ID", "text-analysis-323506")
REGION = os.getenv("GCP_REGION", "us-central1")
GCS_BUCKET = os.getenv("GCS_BUCKET", "text-analysis-323506-artifact-store")

# AI Platform configurations
TRAINING_MACHINE_TYPE = "n1-standard-4"
TRAINING_REPLICA_COUNT = 1
VERTEX_REGION = REGION
SERVING_MACHINE_TYPE = "n1-standard-2"

# Pipeline runtime parameters
TRAINING_STEPS = 1000
EVAL_STEPS = 100

def get_pipeline_config() -> Dict:
    """Returns the pipeline configuration as a dictionary."""
    return {
        "project_id": PROJECT_ID,
        "region": REGION,
        "gcs_bucket": GCS_BUCKET,
        "pipeline_root": PIPELINE_ROOT,
        "pipeline_name": PIPELINE_NAME,
        "namespace": K8S_NAMESPACE,
        "training_machine_type": TRAINING_MACHINE_TYPE,
        "training_replica_count": TRAINING_REPLICA_COUNT,
        "vertex_region": VERTEX_REGION,
        "serving_machine_type": SERVING_MACHINE_TYPE,
        "training_steps": TRAINING_STEPS,
        "eval_steps": EVAL_STEPS,
    }

def get_custom_job_spec() -> Dict:
    """Returns the AI Platform custom job specification."""
    return {
        "project": PROJECT_ID,
        "display_name": f"{PIPELINE_NAME}-training",
        "job_spec": {
            "worker_pool_specs": [{
                "machine_spec": {
                    "machine_type": TRAINING_MACHINE_TYPE,
                },
                "replica_count": TRAINING_REPLICA_COUNT,
                "container_spec": {
                    "image_uri": os.getenv("TFX_IMAGE"),
                }
            }]
        }
    }

def get_vertex_serving_spec() -> Dict:
    """Returns the Vertex AI model serving specification."""
    return {
        "project_id": PROJECT_ID,
        "region": VERTEX_REGION,
        "machine_type": SERVING_MACHINE_TYPE,
        "min_replica_count": 1,
        "max_replica_count": 3,
    } 