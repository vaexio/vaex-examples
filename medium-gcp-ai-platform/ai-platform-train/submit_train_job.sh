# Submit a training job to the GCP AI Platform

# The name of the designated Cloud Storage Bucket
BUCKET=gs://vaex-data

# Name of the job - must be a unique name per project - thus useful to include the data and time of submission
JOB_NAME=har_model_phones_accelerometer_$(date +"%Y_%m_%dT%H_%M_%S")

# The path to a Cloud Storage location at which to store the job's output files
JOB_DIR=$BUCKET/models

# The local path to the root directory of the training application
PACKAGE_PATH=./har_model/

# Specifies the file which the AI Platform should run. This is formated as [training_package.file_to_run.]
MODULE_NAME=har_model.train

# The name of the region at which the job is to be run. Should be the same as the BUCKET region
REGION=europe-west4

# Specify the runtime version. Each version comes pre-installed with specific dependencies.
RUNTIME_VERSION=2.3

# Which python version to run
PYTHON_VERSION=3.7

# Predefined cluster specification. If "CUSTOM", than you need to specify a custom cluster configuration.
SCALE_TIER=CUSTOM

# The master node instance, if the SCALE_TIER is set to "CUSTOM"
MASTER_MACHINE_TYPE=n1-highcpu-32

echo Submitting job $JOB_NAME ...

gcloud ai-platform jobs submit training $JOB_NAME \
    --job-dir=$JOB_DIR \
    --package-path=$PACKAGE_PATH \
    --module-name=$MODULE_NAME \
    --region=$REGION \
    --runtime-version=$RUNTIME_VERSION \
    --python-version=$PYTHON_VERSION \
    --scale-tier=$SCALE_TIER \
    --master-machine-type=$MASTER_MACHINE_TYPE
