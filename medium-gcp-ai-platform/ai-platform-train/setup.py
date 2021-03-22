from setuptools import setup, find_packages


DEPENDENCIES = ['vaex-core==4.1.0',
                'vaex-hdf5==0.7.0',
                'vaex-ml==0.11.1',
                'lightgbm==3.1.1',
                'cloudpickle==1.6.0',
                'gcsfs==0.7.1',
                'astropy==4.1']

setup(name='har_model',
      install_requires=DEPENDENCIES,
      include_package_date=True,
      packages=find_packages(),
      description='Create a Human Activity Recognition model from accelerometer data.')
