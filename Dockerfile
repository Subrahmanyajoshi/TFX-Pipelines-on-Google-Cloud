FROM tensorflow/tfx:1.15.0

# Set working directory
WORKDIR /pipeline

# Install additional dependencies
RUN pip install --no-cache-dir \
    kfp==1.8.22 \
    google-cloud-aiplatform>=1.25.0 \
    google-cloud-storage>=2.10.0 \
    kubernetes>=28.1.0

# Copy pipeline code
COPY pipeline ./pipeline
COPY data ./data

# Set environment variables
ENV PYTHONPATH="/pipeline:${PYTHONPATH}"
ENV KF_PIPELINE_NAME="covertype-training-pipeline"
ENV GOOGLE_APPLICATION_CREDENTIALS="/gcp-credentials/credentials.json"

# Create directory for GCP credentials
RUN mkdir -p /gcp-credentials

# Set the entrypoint
ENTRYPOINT ["python", "/pipeline/runner.py"]
