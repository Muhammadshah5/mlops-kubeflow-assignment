# mlops-kubeflow-assignment
# MLOps Pipeline for California Housing Price Prediction

[![CI/CD](https://github.com/YOUR_USERNAME/mlops-kubeflow-assignment/workflows/ML%20Pipeline%20CI/CD/badge.svg)](https://github.com/YOUR_USERNAME/mlops-kubeflow-assignment/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Minikube Installation](#2-minikube-installation)
  - [3. Kubeflow Pipelines Setup](#3-kubeflow-pipelines-setup)
  - [4. DVC Configuration](#4-dvc-configuration)
  - [5. MLflow Setup](#5-mlflow-setup)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Kubeflow Pipeline](#kubeflow-pipeline)
  - [MLflow Pipeline](#mlflow-pipeline)
- [CI/CD Setup](#cicd-setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

### Problem Statement

This project implements an end-to-end **MLOps pipeline** for predicting California housing prices using machine learning. The pipeline demonstrates industry best practices for:

- **Data Versioning** with DVC
- **Model Training** with Scikit-learn
- **Experiment Tracking** with MLflow
- **Pipeline Orchestration** with Kubeflow Pipelines
- **Continuous Integration** with Jenkins and GitHub Actions

### ML Problem

**Task**: Regression  
**Dataset**: California Housing Dataset (20,640 samples, 8 features)  
**Target**: Median house value for California districts  
**Algorithm**: Random Forest Regressor  

**Features**:
- MedInc: Median income in block group
- HouseAge: Median house age in block group
- AveRooms: Average number of rooms per household
- AveBedrms: Average number of bedrooms per household
- Population: Block group population
- AveOccup: Average number of household members
- Latitude: Block group latitude
- Longitude: Block group longitude

### Key Objectives

✅ Build reproducible ML pipelines  
✅ Version control data and models  
✅ Track experiments and metrics  
✅ Automate testing and deployment  
✅ Demonstrate MLOps best practices  

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         GitHub Repository                        │
│  (Code, Jenkinsfile, Pipeline Definitions, Documentation)       │
└────────────┬────────────────────────────────────────────────────┘
             │
             │ Push/Webhook
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CI/CD (Jenkins/GitHub Actions)                │
│  ├─ Environment Setup                                            │
│  ├─ Code Quality Checks                                          │
│  ├─ Pipeline Compilation                                         │
│  ├─ Unit Tests                                                   │
│  └─ Artifact Archival                                            │
└────────────┬────────────────────────────────────────────────────┘
             │
             │ Deploy
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Kubeflow Pipelines (Minikube)                 │
│                                                                   │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐ │
│  │     Data     │───▶│  Preprocessing   │───▶│    Model      │ │
│  │  Extraction  │    │   & Splitting    │    │   Training    │ │
│  └──────────────┘    └──────────────────┘    └───────┬───────┘ │
│                                                        │         │
│                                               ┌────────▼───────┐ │
│                                               │     Model      │ │
│                                               │   Evaluation   │ │
│                                               └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    MLflow Tracking Server                        │
│  ├─ Experiment Tracking                                          │
│  ├─ Metrics & Parameters                                         │
│  ├─ Model Registry                                               │
│  └─ Artifact Storage                                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    DVC Remote Storage                            │
│  (Google Drive / S3 / Azure Blob / Local)                        │
│  ├─ Raw Data                                                     │
│  ├─ Processed Data                                               │
│  └─ Model Artifacts                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technologies Used

| Category | Technology | Purpose |
|----------|-----------|---------|
| **ML Framework** | Scikit-learn | Model training and evaluation |
| **Orchestration** | Kubeflow Pipelines | Pipeline automation |
| **Experiment Tracking** | MLflow | Tracking experiments and models |
| **Data Versioning** | DVC | Version control for data |
| **Container Platform** | Minikube | Local Kubernetes cluster |
| **CI/CD** | Jenkins, GitHub Actions | Automated testing and deployment |
| **Programming** | Python 3.10 | Core development language |
| **Storage** | MinIO, Google Drive | Artifact and data storage |

---

## 📦 Prerequisites

Before starting, ensure you have the following installed:

- **Python** 3.10 or higher
- **Docker** (for Jenkins and Minikube)
- **Minikube** (for Kubernetes cluster)
- **kubectl** (Kubernetes CLI)
- **Git** (version control)
- **pip** (Python package manager)

### System Requirements

- **OS**: macOS, Linux, or Windows with WSL2
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk Space**: 40GB free
- **CPU**: 4 cores recommended

---

## 🚀 Setup Instructions

### 1. Environment Setup

#### Clone Repository

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mlops-kubeflow-assignment.git
cd mlops-kubeflow-assignment

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Install Core Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installations
python -c "import kfp; print(f'KFP version: {kfp.__version__}')"
python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')"
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"
```

---

### 2. Minikube Installation

#### Install Minikube

**macOS:**
```bash
brew install minikube
```

**Linux:**
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

**Windows:**
```powershell
choco install minikube
```

#### Start Minikube Cluster

```bash
# Start Minikube with adequate resources
minikube start \
  --cpus=4 \
  --memory=8192 \
  --disk-size=40g \
  --driver=docker

# Verify cluster is running
minikube status

# Expected output:
# minikube
# type: Control Plane
# host: Running
# kubelet: Running
# apiserver: Running
```

#### Install kubectl

```bash
# macOS
brew install kubectl

# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Verify
kubectl version --client
```

---

### 3. Kubeflow Pipelines Setup

#### Deploy Kubeflow Pipelines

```bash
# Set KFP version
export PIPELINE_VERSION=1.8.5

# Deploy cluster-scoped resources
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"

# Wait for CRDs to be established
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

# Deploy Kubeflow Pipelines
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"

# Wait for all pods to be ready (10-15 minutes)
kubectl wait --for=condition=ready pod --all -n kubeflow --timeout=900s
```

#### Verify Installation

```bash
# Check all pods are running
kubectl get pods -n kubeflow

# All pods should show STATUS: Running
```

#### Access Kubeflow Pipelines UI

```bash
# Port forward to access UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

# Open in browser: http://localhost:8080
```

---

### 4. DVC Configuration

#### Install DVC

```bash
# Install DVC with remote storage support
pip install 'dvc[gdrive]'  # For Google Drive
# OR
pip install 'dvc[s3]'      # For AWS S3
# OR
pip install 'dvc[azure]'   # For Azure Blob Storage
```

#### Initialize DVC

```bash
# Initialize DVC in project
dvc init

# Add remote storage (example: Google Drive)
dvc remote add -d myremote gdrive://YOUR_GOOGLE_DRIVE_FOLDER_ID

# OR for local remote (for testing)
dvc remote add -d myremote /path/to/dvc/storage

# Configure remote
dvc remote modify myremote gdrive_use_service_account true
```

#### Track Data with DVC

```bash
# Add data to DVC tracking
dvc add data/raw/raw_data.csv

# Commit DVC file
git add data/raw/raw_data.csv.dvc .gitignore
git commit -m "Track raw data with DVC"

# Push data to remote
dvc push

# Pull data (when needed)
dvc pull
```

---

### 5. MLflow Setup

#### Start MLflow Tracking Server

```bash
# Option 1: Simple file-based tracking
mlflow ui --port 5000

# Option 2: With backend database (recommended for production)
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000

# Access MLflow UI: http://localhost:5000
```

#### Configure MLflow in Code

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("California Housing Pipeline")
```

---

## 📊 Pipeline Walkthrough

### Kubeflow Pipeline

#### Pipeline Components

The Kubeflow pipeline consists of 4 main components:

1. **Data Extraction** (`data_extraction`)
   - Fetches California Housing dataset
   - Saves to CSV format
   - Logs dataset metadata

2. **Data Preprocessing** (`data_preprocessing`)
   - Splits data into train/test sets (80/20)
   - Handles missing values
   - Logs preprocessing parameters

3. **Model Training** (`model_training`)
   - Trains Random Forest Regressor
   - Logs hyperparameters
   - Saves trained model

4. **Model Evaluation** (`model_evaluation`)
   - Evaluates on test set
   - Computes metrics (MSE, RMSE, MAE, R²)
   - Logs metrics and artifacts

#### Compile Pipeline

```bash
# Compile Kubeflow pipeline
python pipeline.py

# This generates: pipeline.yaml
```

#### Upload to Kubeflow

**Method 1: Via UI**
1. Open Kubeflow UI: http://localhost:8080
2. Click **Pipelines** → **Upload pipeline**
3. Select `pipeline.yaml`
4. Name: `California Housing Pipeline`
5. Click **Create**

**Method 2: Via Python SDK**
```python
import kfp

client = kfp.Client(host='http://localhost:8080')

# Upload pipeline
client.upload_pipeline(
    pipeline_package_path='pipeline.yaml',
    pipeline_name='California Housing Pipeline'
)

# Create experiment
experiment = client.create_experiment(name='Housing Experiments')

# Run pipeline
run = client.run_pipeline(
    experiment_id=experiment.id,
    job_name='housing-run-1',
    pipeline_package_path='pipeline.yaml'
)

print(f"Pipeline run started: {run.id}")
```

#### Monitor Pipeline Execution

```bash
# Watch pipeline progress
kubectl get pods -n kubeflow -w

# View logs of specific component
kubectl logs -n kubeflow <pod-name>

# Check workflow status
kubectl get workflow -n kubeflow
```

---

### MLflow Pipeline

#### Run MLflow Pipeline

```bash
# Navigate to MLflow pipeline directory
cd mlflow_pipeline

# Ensure MLflow server is running (in another terminal)
mlflow ui --port 5000

# Run the pipeline
python pipeline_mlflow.py
```

#### Pipeline Output

```
======================================================================
🚀 CALIFORNIA HOUSING ML PIPELINE
======================================================================

📦 STEP 1: Data Extraction
----------------------------------------------------------------------
✅ Dataset loaded: 20640 samples, 9 features

🔧 STEP 2: Data Preprocessing
----------------------------------------------------------------------
✅ Training set: 16512 samples
✅ Test set: 4128 samples

🤖 STEP 3: Model Training
----------------------------------------------------------------------
🌲 Training Random Forest (trees=100, depth=10)...
✅ Model trained successfully

📊 STEP 4: Model Evaluation
----------------------------------------------------------------------
📈 Evaluation Metrics:
   MSE:  0.2543
   RMSE: 0.5043
   MAE:  0.3312
   R²:   0.8142
✅ Evaluation complete

======================================================================
✅ PIPELINE COMPLETED SUCCESSFULLY!