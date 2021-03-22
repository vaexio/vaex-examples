import os
import logging
from datetime import datetime

import numpy as np

from sklearn.metrics import accuracy_score, f1_score

import vaex
import vaex.ml
import vaex.ml.lightgbm

# Configure logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

# Stream the data from GCS
log.info('Reading in the data...')
df = vaex.open('gs://vaex-data/human-activity-recognition/phones_accelerometer.hdf5')

# Pre-fetch only the relevant column
log.info('Fetching relevant columns...')
columns_to_use = ['Arrival_Time', 'Creation_Time', 'x', 'y', 'z', 'gt']
df.nop(columns_to_use)

# Train and test split
log.info('Splitting the data into train, validation and test sets...')
df_train, df_val, df_test = df.split_random(into=[0.8, 0.1, 0.1], random_state=42)

# Feature engineering
log.info('Feature engineering...')

# Drop missing values form unlabeled activities
log.info('Drop missing values form unlabeled activities...')
df_train = df_train.dropna(column_names=['gt'])

log.info('Convert to spherical polar coordinates...')
df_train['r'] = ((df_train.x**2 + df_train.y**2 + df_train.z**2)**0.5).jit_numba()
df_train['theta'] = np.arccos(df_train.z / df_train.r).jit_numba()
df_train['phi'] = np.arctan2(df_train.y, df_train.x).jit_numba()

log.info('PCA transformation...')
df_train = df_train.ml.pca(n_components=3, features=['x', 'y', 'z'])

log.info('Create certain feature interactions...')
df_train['PCA_00'] = df_train.PCA_0**2
df_train['PCA_11'] = df_train.PCA_1**2
df_train['PCA_22'] = df_train.PCA_2**2

df_train['PCA_01'] = df_train.PCA_0 * df_train.PCA_1
df_train['PCA_02'] = df_train.PCA_0 * df_train.PCA_2
df_train['PCA_12'] = df_train.PCA_1 * df_train.PCA_2

log.info('Calculate some summary statistics per class...')
df_summary = df_train.groupby('gt').agg({'PCA_0_mean': vaex.agg.mean('PCA_0'),
                                         'PCA_0_std': vaex.agg.std('PCA_0'),
                                         'PCA_1_mean': vaex.agg.mean('PCA_1'),
                                         'PCA_1_std': vaex.agg.std('PCA_1'),
                                         'PCA_2_mean': vaex.agg.mean('PCA_2'),
                                         'PCA_2_std': vaex.agg.std('PCA_2')
                                         }).to_pandas_df().set_index('gt')

log.info('Define features based on the summary statistics per target class...')
for class_name in df_train.gt.unique():
    feature_name = f'PCA_012_err_{class_name}'
    df_train[feature_name] = ((np.abs(df_train.PCA_0 - df_summary.loc[class_name, 'PCA_0_mean']) / df_summary.loc[class_name, 'PCA_0_std']) +
                              (np.abs(df_train.PCA_1 - df_summary.loc[class_name, 'PCA_1_mean']) / df_summary.loc[class_name, 'PCA_1_std']) +
                              (np.abs(df_train.PCA_2 - df_summary.loc[class_name, 'PCA_2_mean']) / df_summary.loc[class_name, 'PCA_2_std'])).jit_numba()

log.info('Create features based on KMeans clustering...')
n_clusters = df_train.gt.nunique()

logging.info('Creating kmeans clustering features using the PCA components ...')
df_train = df_train.ml.kmeans(features=['PCA_0', 'PCA_1', 'PCA_2'],
                              n_clusters=n_clusters,
                              max_iter=1000,
                              n_init=5,
                              prediction_label='kmeans_pca')

logging.info('Creating kmeans clustering features using the interacting PCA components ...')
df_train = df_train.ml.kmeans(features=['PCA_01', 'PCA_02', 'PCA_12'],
                              n_clusters=n_clusters,
                              max_iter=1000,
                              n_init=5,
                              prediction_label='kmeans_pca_inter')

logging.info('Creating kmeans clustering features using the power PCA components ...')
df_train = df_train.ml.kmeans(features=['PCA_00', 'PCA_11', 'PCA_22'],
                              n_clusters=n_clusters,
                              max_iter=1000,
                              n_init=5,
                              prediction_label='kmeans_pca_power')

log.info('Create time feature...')
df_train['time_delta'] = df_train.Arrival_Time - df_train.Creation_Time
df_train = df_train.ml.max_abs_scaler(features=['time_delta'], prefix='scaled_')

log.info('Gather all the features that will be used for training the model...')
features = df_train.get_column_names(regex='x|y|z|r|theta|phi|PCA_|scaled_|kmeans_')

log.info('Encoding the target variable...')
target_encoder = df_train.ml.label_encoder(features=['gt'], prefix='enc_', transform=False)
df_train = target_encoder.transform(df_train)
target_mapper_inv = {key: value for value, key in target_encoder.labels_['gt'].items()}

# Apply the feature transformations to the validation set
# so it can be used while fitting the estimator
log.info('Applying the feature transformations on the validation set...')
df_val.state_set(df_train.state_get())

# Training the Estimator
log.info('Instantiating and configuring a LightGBM model...')

# Train a lightgbm model
params = {
    'learning_rate': 0.5,     # learning rate
    'max_depth': 7,           # max depth of the tree
    'colsample_bytree': 0.8,  # subsample ratio of columns when constructing each tree
    'subsample': 0.8,         # subsample ratio of the training instance
    'reg_lambda': 3,          # L2 regularisation
    'reg_alpha': 1.5,         # L1 regularisation
    'min_child_weight': 1,    # minimum sum of instance weight (hessian) needed in a child
    'objective': 'softmax',   # learning task objective
    'num_class': 6,           # number of target classes (if classification)
    'random_state': 42,       # fixes the seed, for reproducibility
    'metric': 'multi_error'   # the error metric
}

# Instantiate the booster model
booster = vaex.ml.lightgbm.LightGBMModel(features=features,
                                         target='enc_gt',
                                         prediction_name='pred',
                                         num_boost_round=1000,
                                         params=params)

history = {}  # Dict in which to record the training history
# Start the training process
log.info('Training the LightGBM model...')
booster.fit(df=df_train,
            valid_sets=[df_train, df_val],
            valid_names=['train', 'val'],
            early_stopping_rounds=15,
            evals_result=history,
            verbose_eval=True)

log.info('Obtain predictions for the training set...')
df_train = booster.transform(df_train)

log.info('Get the names of the predicted classes...')
df_train['pred_name'] = df_train.pred.apply(lambda x: target_mapper_inv[np.argmax(x)])

# Model evaluation
log.info('Evaluating the trained model...')

# Apply the full pipeline to the validation and test samples
df_test.state_set(df_train.state_get())
df_val.state_set(df_train.state_get())

# Get the scores
val_acc = accuracy_score(df_val.pred.values.argmax(axis=1), df_val.enc_gt.values)
test_acc = accuracy_score(df_test.pred.values.argmax(axis=1), df_test.enc_gt.values)

val_f1 = f1_score(df_val.pred.values.argmax(axis=1), df_val.enc_gt.values, average='micro')
test_f1 = f1_score(df_test.pred.values.argmax(axis=1), df_test.enc_gt.values, average='micro')

log.info('Evaluating the model performance...')
log.info(f'Validation accuracy: {val_acc:.3f}')
log.info(f'Validation f1-score: {val_f1:.3f}')
log.info(f'Test accuracy: {test_acc:.3f}')
log.info(f'Test f1-score: {test_f1:.3f}')

# Save the model to a GCP bucket - vaex can do this directly!
log.info('Saving the Vaex state file to a GCP bucket...')
bucket_name = 'gs://vaex-data'
folder_name = datetime.now().strftime('models/har_phones_accelerometer_%Y-%m-%dT%H:%M:%S')
model_name = 'state.json'
gcs_model_path = os.path.join(bucket_name, folder_name, model_name)
# Save only the columns that are needed in production
df_train[features + ['pred', 'pred_name']].state_write(gcs_model_path)

log.info(f'The model has been trained and is available in {bucket_name}.')
# THE END