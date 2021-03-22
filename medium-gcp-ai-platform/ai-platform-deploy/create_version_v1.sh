# The model name
MODEL=har

# Version name
VERSION=v1

# The region
REGION=global

# Specify the runtime version. Each version comes pre-installed with specific dependencies.
RUNTIME_VERSION=2.3

# Which python version to run
PYTHON_VERSION=3.7

# Path to the directory containing the model state file
ORIGIN=gs://vaex-data/models/har_phones_accelerometer_2021-03-19T21:37:37

# Path to prediction module
PREDICTION_PACKAGE_PATH=gs://vaex-data/deployments/vaex_predictor-0.0.0.tar.gz

# The prediction class, located within the prediction module
PREDICTION_CLASS=predictor.VaexPredictor

# # The master node instance, if the SCALE_TIER is set to "CUSTOM"
# MASTER_MACHINE_TYPE=n1-highcpu-32

echo Creating version $VERSION of model $MODEL ...

gcloud beta ai-platform versions create $VERSION \
    --model=$MODEL \
    --region=$REGION \
    --runtime-version=$RUNTIME_VERSION \
    --python-version=$PYTHON_VERSION \
    --origin=$ORIGIN \
    --package-uris=$PREDICTION_PACKAGE_PATH \
    --prediction-class=$PREDICTION_CLASS
    # --machine-type=$MASTER_MACHINE_TYPE
