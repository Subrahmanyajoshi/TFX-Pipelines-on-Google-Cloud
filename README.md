# TFX-Pipelines-on-Google-Cloud
This repository shows how to create and deploy TFX (TensorFlow Extended) pipelines on Google Cloud Platform.
It includes examples of:

- Building a TFX pipeline using the TFX Python API.
- Configuring TFX components to run on Google Cloud services like:
    - **Cloud Storage:** For artifact storage.
    - **AI Platform Training:** For distributed model training and hyperparameter tuning (Trainer, Tuner).
    - **Kubeflow Pipelines (KFP):** As the orchestrator to run the TFX pipeline on Google Kubernetes Engine (GKE).
- Using TFX components like ExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Tuner, Evaluator, InfraValidator, and Pusher.
- Implementing custom preprocessing and model code (`preprocessing.py`, `model.py`).
- Configuring runtime parameters for the pipeline.
- Setting up a configuration file (`config.py`) to manage environment-specific settings.

## Project Structure

```
.
├── data/
│   └── covertype_dataset.csv  # Sample dataset
├── pipeline/
│   ├── config.py              # Configuration settings
│   ├── model.py               # Model definition and training code
│   ├── pipeline.py            # TFX pipeline definition
│   ├── preprocessing.py       # Data preprocessing code
│   └── runner.py             # Script to build and run the pipeline
├── k8s/
│   └── pipeline.yaml         # Kubernetes manifests for pipeline deployment
├── Dockerfile               # Container definition for pipeline components
├── deploy_to_gcp.py        # Deployment script for Google Cloud
└── README.md
```

## Prerequisites

1. Install the Google Cloud SDK: [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)
2. Install kubectl: [https://kubernetes.io/docs/tasks/tools/](https://kubernetes.io/docs/tasks/tools/)
3. Python 3.7 or later
4. A Google Cloud project with billing enabled

## Configuration

1. Update `pipeline/config.py` with your Google Cloud project settings:
   - PROJECT_ID: Your Google Cloud project ID
   - GCP_REGION: Your preferred Google Cloud region
   - ARTIFACT_STORE_URI: GCS bucket for storing pipeline artifacts
   - CUSTOM_SERVICE_ACCOUNT: Service account email (will be created during deployment)

2. (Optional) Modify the machine types and other configurations in:
   - `k8s/pipeline.yaml` for Kubernetes resources
   - `pipeline/kubeflow_config.py` for Kubeflow-specific settings

## Deployment

The project includes a deployment script that automates the setup process:

```bash
# Install required Python packages
pip install -r requirements.txt

# Deploy to Google Cloud
python deploy_to_gcp.py --project-id=YOUR_PROJECT_ID [--region=YOUR_REGION] [--cluster-name=YOUR_CLUSTER_NAME]
```

The deployment script will:
1. Enable necessary Google Cloud APIs
2. Create a GKE cluster
3. Install Kubeflow
4. Set up workload identity
5. Build and push the pipeline container image
6. Deploy the pipeline to Kubeflow

## Accessing the Pipeline

After deployment, you can access your pipeline through the Kubeflow Pipelines UI:

1. Get the external IP of the Kubeflow gateway:
```bash
kubectl get svc -n kubeflow istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

2. Open the IP address in your browser
3. Navigate to the Pipelines section
4. Your pipeline will be listed under the name specified in `config.py`

## Running the Pipeline

You can run the pipeline either through:

1. The Kubeflow Pipelines UI:
   - Click on your pipeline
   - Click "Create Run"
   - Configure any runtime parameters
   - Click "Start"

2. The command line:
```bash
python pipeline/runner.py
```

## Monitoring and Debugging

1. View pipeline runs in the Kubeflow Pipelines UI
2. Check component logs in the UI or using kubectl:
```bash
kubectl logs -n kubeflow -l app=tfx-pipeline
```

3. Monitor AI Platform jobs in the Google Cloud Console
4. View pipeline artifacts in Cloud Storage

## Cleanup

To clean up resources when you're done:

1. Delete the GKE cluster:
```bash
gcloud container clusters delete CLUSTER_NAME --region=REGION
```

2. Delete the GCS bucket with artifacts:
```bash
gsutil rm -r gs://YOUR_BUCKET_NAME
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
