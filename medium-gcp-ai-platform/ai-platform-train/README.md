# Train a model on the "classical" AI Platform

### Train locally

```bash
gcloud ai-platform local train \
    --package-path="./har_model/" \
    --module-name="har_model.train"
```

### Train on the AI Platform

Specify training job parameters

```bash
BUCKET=gs://vaex-data
JOB_NAME=har_model_$(date +"%Y%m%d_%H%M%S")
JOB_DIR=$BUCKET/models
PACKAGE_PATH=./har_model/
MODULE_NAME=har_model.train
REGION=europe-west4
RUNTIME_VERSION=2.3
PYTHON_VERSION=3.7
SCALE_TIER=CUSTOM
MASTER_MACHINE_TYPE=n1-highcpu-32


gcloud ai-platform jobs submit training $JOB_NAME \
    --job-dir=$JOB_DIR \
    --package-path=$PACKAGE_PATH \
    --module-name=$MODULE_NAME \
    --region=$REGION \
    --runtime-version=$RUNTIME_VERSION \
    --python-version=$PYTHON_VERSION \
    --scale-tier=$SCALE_TIER \
    --master-machine-type=$MASTER_MACHINE_TYPE
```

# Train on the Unified AI Platform

### Using a predefined container

Create and upload the model trained package to GCS

```bash
BUCKET=gs://vaex-data

bash make_package.sh
```
Submit the training job

```bash
REGION=europe-west4
JOB_NAME=har_model_$(date +"%Y%m%d_%H%M%S")
PYTHON_PACKAGE_IMAGE_URI=europe-docker.pkg.dev/cloud-aiplatform/training/scikit-learn-cpu.0-23:latest
PYTHON_PACKAGE_URIS=gs://vaex-data/training-modules/har_model-0.0.0.tar.gz
MODULE_NAME=har_model.train
MASTER_MACHINE_TYPE=n1-highcpu-32

gcloud beta ai custom-jobs create \
  --region=$REGION \
  --display-name=$JOB_NAME \
  --python-package-uris=$PYTHON_PACKAGE_URIS \
  --worker-pool-spec=machine-type=$MASTER_MACHINE_TYPE,replica-count=1,python-image-uri=$PYTHON_PACKAGE_IMAGE_URI,python-module=$MODULE_NAME
```

# Using a custom container

First create a _Dockerfile_ containing the package that you want to use for training
and all of its dependencies. When launched, the container should start the training job.

```bash
JOB_NAME=har_model_$(date +"%Y%m%d_%H%M%S")
REGION=europe-west4
PROJECTID=
IMAGE_NAME=train-image
CUSTOM_CONTAINER_IMAGE_URI=gcr.io/$PROJECTID/$IMAGE_NAME
MASTER_MACHINE_TYPE=n1-highcpu-32
```

Build the container on GCP

```bash
gcloud builds submit --tag $CUSTOM_CONTAINER_IMAGE_URI
```

Start the training job

```bash
gcloud beta ai custom-jobs create \
  --region=$REGION \
  --display-name=$JOB_NAME \
  --worker-pool-spec=machine-type=$MASTER_MACHINE_TYPE,replica-count=1,container-image-uri=$CUSTOM_CONTAINER_IMAGE_URI
```