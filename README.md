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
│   └── runner.py              # Script to build and run the pipeline
├── README.md
