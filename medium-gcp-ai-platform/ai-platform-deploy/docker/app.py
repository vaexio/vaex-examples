import os

import uvicorn
from fastapi import FastAPI
from fastapi.logger import logger
from pydantic import BaseModel

import numpy as np

import vaex


# Instantiate the web application
app = FastAPI()

class Data(BaseModel):
    instances: list = []
    parameters: dict

# These will be some global parameters, essentially filling them up at startup
global_items = {}

# Run at startup - basically downloads the state file
@app.on_event("startup")
def startup():
    path = os.getenv('AIP_STORAGE_URI')
    fs_options = {'token': 'cloud'}

    if path is not None:
        logger.info(f'Loading model state from: {path}')
        path = os.path.join(path, 'state.json')
        global_items['state'] = vaex.utils.read_json_or_yaml(path, fs_options=fs_options)

    else:  # for local testing
        logger.info(f'AIP_STORAGE_URI was not set - loading state from a default location')
        path ='gs://vaex-data/models/har_phones_accelerometer_2021-03-13T14:10:54/state.json'
        global_items['state'] = vaex.utils.read_json_or_yaml(path, fs_options=fs_options)

    logger.info('State successfully retrieved from GCP')


# Healthcheck
@app.get('/health')
def health():
    return ''


@app.post('/predict')
def predict(data: Data):
    instances = data.instances

    if isinstance(instances[0], list):
        data = np.asarray(instances).T
        df = vaex.from_arrays(Arrival_Time=data[0],
                              Creation_Time=data[1],
                              x=data[2],
                              y=data[3],
                              z=data[4])

    elif isinstance(instances[0], dict):
        dfs = []
        for instance in instances:
            df = vaex.from_dict(instance)
            dfs.append(df)
        df = vaex.concat(dfs)

    else:
        return {'predictions': 'invalid input format'}

    df.state_set(global_items['state'], set_filter=False)
    return {'predictions': df.pred_name.tolist()}


if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=False)
