# How to deploy a model on the "classical" AI Platform

### Specify environmental variables

```
BUCKET=gs://vaex-data
MODEL=har
REGION=global
VERSION=v1
RUNTIME_VERSION=2.3
PYTHON_VERSION=3.7
ORIGIN=gs://vaex-data/models/har_phones_accelerometer_2021-03-22T19:15:06
PREDICTION_PACKAGE_PATH=gs://vaex-data/deployments/vaex_predictor-0.0.0.tar.gz
PREDICTION_CLASS=predictor.VaexPredictor
```

### Create a Python package out of the vaex_predictor module
`python setup.py sdist --formats=gztar`

### Copy the package Google Cloud Storage
`gsutil cp dist/vaex_predictor-0.0.0.tar.gz $BUCKET/deployments/`

### Create a \_model\_ on the AI Platform
```
gcloud beta ai-platform models create $MODEL \
  --region=$REGION \
  --enable-logging \
  --enable-console-logging
```

### Create a model \_version\_:
```
gcloud beta ai-platform versions create $VERSION \
    --model=$MODEL \
    --region=$REGION \
    --runtime-version=$RUNTIME_VERSION \
    --python-version=$PYTHON_VERSION \
    --origin=$ORIGIN \
    --package-uris=$PREDICTION_PACKAGE_PATH \
    --prediction-class=$PREDICTION_CLASS
```

### Send online prediction requests
`gcloud ai-platform predict --model=$MODEL --version=$VERSION --json-instances=input_dict.json --region=$REGION`
`gcloud ai-platform predict --model=$MODEL --version=$VERSION --json-instances=input_list.json --region=$REGION`

### Delete a model version
`gcloud ai-platform versions delete $VERSION --model=$MODEL --region=$REGION`

### Delete the model
`gcloud ai-platform models delete $MODEL --region=$REGION`

# How to deploy a model on the Unified AI Platform using a predefined container

This is not supported.


# How to deploy a model on the Unified AI Platform using a custom container


### Create an image on Container Registry

PROJECTID=
IMAGE_NAME=predict-image
CUSTOM_CONTAINER_IMAGE_URI=gcr.io/$PROJECTID/$IMAGE_NAME

```
gcloud builds submit --tag $CUSTOM_CONTAINER_IMAGE_URI
```

### Upload a model to the platform - in this case it is a container

REGION=europe-west4
MODEL_NAME=har_model
PATH_TO_STATE_FILE=gs://vaex-data-europe-west4/models/har_phones_accelerometer_2021-03-22T19:15:06


```
gcloud beta ai models upload \
  --region=$REGION \
  --display-name=$MODEL_NAME \
  --container-image-uri=$CUSTOM_CONTAINER_IMAGE_URI \
  --artifact-uri=$PATH_TO_STATE_FILE \
  --container-health-route=/health \
  --container-ports=8000 \
  --container-predict-route=/predict
```

### Create an endpoint

ENDPOINT_NAME=har_model_endpoint

```
gcloud beta ai endpoints create \
  --region=$REGION \
  --display-name=$ENDPOINT_NAME
```

### Get the model and endpoint IDs

These can be found on the Google Cloud Console once the resources are up and running.

MODEL_ID=
ENDPOINT_ID=

### Deploy the model

```
gcloud beta ai endpoints deploy-model $ENDPOINT_ID \
  --region=$REGION \
  --model=$MODEL_ID \
  --display-name=deployed_har_model \
  --machine-type=n1-standard-2 \
  --min-replica-count=1 \
  --max-replica-count=1 \
  --traffic-split=0=100 \
  --enable-access-logging \
  --enable-container-logging
```

### Send a request

INPUT_DATA_FILE=input_json.json

```
curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json" https://${REGION}-aiplatform.googleapis.com/v1alpha1/projects/${PROJECT_ID}/locations/us-central1/endpoints/${ENDPOINT_ID}:predict -d "@${INPUT_DATA_FILE}"
```

### Undeploy the model

Find the deployed model id with this command
`gcloud beta ai models list --region=$REGION | grep deployed`

DEPLOYED_MODEL_ID=

```
gcloud beta ai endpoints undeploy-model $ENDPOINT_ID \
  --project=$PROJECTID  \
  --region=$REGION \
  --deployed-model-id=$DEPLOYED_MODEL_ID
```

### Delete the endpoint

`gcloud beta ai endpoints delete $ENDPOINT_ID --region=$REGION`

### Delete the model

`gcloud beta ai models delete $MODEL_ID --region=$REGION`