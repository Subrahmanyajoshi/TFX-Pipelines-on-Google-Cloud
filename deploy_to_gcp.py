#!/usr/bin/env python3
"""Script to deploy the TFX pipeline to Google Cloud Platform."""

import argparse
import os
import subprocess
from typing import List, Optional

def run_command(cmd: List[str], cwd: Optional[str] = None) -> None:
    """Run a shell command."""
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)

def setup_gcp_project(project_id: str, region: str) -> None:
    """Set up GCP project and enable necessary APIs."""
    apis_to_enable = [
        "container.googleapis.com",      # Google Kubernetes Engine API
        "cloudbuild.googleapis.com",     # Cloud Build API
        "artifactregistry.googleapis.com", # Artifact Registry API
        "aiplatform.googleapis.com",     # Vertex AI API
        "dataflow.googleapis.com",       # Dataflow API
        "iam.googleapis.com"             # Identity and Access Management API
    ]
    
    print(f"\nSetting up GCP project {project_id}...")
    
    # Set default project
    run_command(["gcloud", "config", "set", "project", project_id])
    
    # Enable required APIs
    for api in apis_to_enable:
        run_command(["gcloud", "services", "enable", api])

def create_gke_cluster(project_id: str, region: str, cluster_name: str) -> None:
    """Create a GKE cluster for Kubeflow."""
    print(f"\nCreating GKE cluster {cluster_name}...")
    
    run_command([
        "gcloud", "container", "clusters", "create", cluster_name,
        f"--project={project_id}",
        f"--region={region}",
        "--machine-type=n1-standard-4",
        "--num-nodes=3",
        "--workload-pool={project_id}.svc.id.goog",
        "--enable-ip-alias",
        "--enable-workload-identity"
    ])

def install_kubeflow(cluster_name: str, region: str) -> None:
    """Install Kubeflow on the GKE cluster."""
    print("\nInstalling Kubeflow...")
    
    # Get cluster credentials
    run_command([
        "gcloud", "container", "clusters", "get-credentials",
        cluster_name, f"--region={region}"
    ])
    
    # Create kubeflow namespace
    run_command(["kubectl", "create", "namespace", "kubeflow"])

def setup_workload_identity(project_id: str) -> None:
    """Set up workload identity for the pipeline."""
    print("\nSetting up workload identity...")
    
    # Create service account
    sa_name = "pipeline-runner"
    sa_email = f"{sa_name}@{project_id}.iam.gserviceaccount.com"
    
    run_command([
        "gcloud", "iam", "service-accounts", "create", sa_name,
        "--display-name=Pipeline Runner"
    ])
    
    # Grant necessary roles
    roles = [
        "roles/storage.admin",
        "roles/aiplatform.user",
        "roles/dataflow.developer",
        "roles/iam.serviceAccountUser"
    ]
    
    for role in roles:
        run_command([
            "gcloud", "projects", "add-iam-policy-binding", project_id,
            f"--member=serviceAccount:{sa_email}",
            f"--role={role}"
        ])
    
    # Bind KSA to GSA
    run_command([
        "gcloud", "iam", "service-accounts", "add-iam-policy-binding",
        sa_email,
        f"--member=serviceAccount:{project_id}.svc.id.goog[kubeflow/{sa_name}]",
        "--role=roles/iam.workloadIdentityUser"
    ])

def build_and_push_image(project_id: str) -> None:
    """Build and push the pipeline container image."""
    print("\nBuilding and pushing container image...")
    
    image_name = f"gcr.io/{project_id}/tfx-pipeline:latest"
    
    run_command([
        "gcloud", "builds", "submit",
        f"--tag={image_name}",
        "."
    ])

def deploy_pipeline(project_id: str) -> None:
    """Deploy the Kubernetes resources for the pipeline."""
    print("\nDeploying pipeline resources...")
    
    # Create ConfigMap for GCP configuration
    run_command([
        "kubectl", "create", "configmap", "gcp-config",
        f"--from-literal=project-id={project_id}",
        f"--from-literal=region={args.region}",
        "-n", "kubeflow"
    ])
    
    # Apply Kubernetes manifests
    run_command(["kubectl", "apply", "-f", "k8s/pipeline.yaml"])

def main(args: argparse.Namespace) -> None:
    """Main deployment function."""
    setup_gcp_project(args.project_id, args.region)
    create_gke_cluster(args.project_id, args.region, args.cluster_name)
    install_kubeflow(args.cluster_name, args.region)
    setup_workload_identity(args.project_id)
    build_and_push_image(args.project_id)
    deploy_pipeline(args.project_id)
    
    print("\nDeployment completed successfully!")
    print(f"\nTo access your pipeline, get the external IP of the Kubeflow gateway:")
    print(f"kubectl get svc -n kubeflow istio-ingressgateway -o jsonpath='{{.status.loadBalancer.ingress[0].ip}}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy TFX pipeline to Google Cloud")
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument("--region", default="us-central1", help="Google Cloud region")
    parser.add_argument("--cluster-name", default="kubeflow-cluster", help="GKE cluster name")
    
    args = parser.parse_args()
    main(args) 