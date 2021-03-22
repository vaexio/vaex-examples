import os
os.environ['HOME'] = '/tmp'

import json

import numpy as np

import vaex


class VaexPredictor():
    def __init__(self, state=None):
        self.state = state

    def predict(self, instances, **kwargs):

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
            return ['invalid input format']

        df.state_set(self.state, set_filter=False)
        return df.pred_name.tolist()

    @classmethod
    def from_path(cls, model_dir):
        model_path = os.path.join(model_dir, 'state.json')
        with open(model_path) as f:
            state = json.load(f)

        return cls(state)
