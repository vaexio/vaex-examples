import os
from setuptools import setup, find_packages


os.environ['HOME'] = '/tmp'

DEPENDENCIES = ['vaex-core==4.1.0',
                'vaex-ml==0.11.1',
                'lightgbm==3.1.1',
                'cloudpickle==1.6.0',
                'gcsfs==0.7.1',
                'astropy==4.1']

setup(
    name='vaex_predictor',
    scripts=['predictor.py'],
    install_requires=DEPENDENCIES,
    include_package_date=True,
    packages=find_packages(),
    description='Prediction routine for the Human Activity Recognition phone accelerometer model.')
