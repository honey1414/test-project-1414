import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import gmean
from scipy.stats import hmean
from scipy.stats import moment
from scipy.stats import variation

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from lightgbm import LGBMRegressor

from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import NMF

from sklearn.feature_selection import f_regression
from sklearn.model_selection import RandomizedSearchCV

from tqdm import tqdm

tqdm.pandas()

import joblib


# functions for preprocessing
def preprocessing(data):
    label = data['target']
    features = data.drop(['ID', 'target'], axis=1)

    # applying the log-transformation
    features = np.log1p(features)
    label = np.log1p(label)

    # removing the constant features
    cols_with_one_value = features.columns[features.nunique() == 1]
    features.drop(cols_with_one_value, axis=1, inplace=True)

    # feature decomposition using PCA
    pca = PCA(n_components=50)
    features_PCA = pca.fit_transform(features)

    # feature decomposition using SRP
    srp = SparseRandomProjection(n_components=100, eps=0.28, dense_output=False)
    features_SRP = srp.fit_transform(features)

    # feature decomposition using NMF
    nmf = NMF(n_components=100, init=None, solver="cd", beta_loss="frobenius",
              tol=0.0001, max_iter=200, random_state=None, alpha=0.0,
              l1_ratio=0.0, verbose=0, shuffle=False)
    features_NMF = nmf.fit_transform(features)

    # concatenating the decomposed features all together
    features_PCA = features_PCA[:, :20]
    features_SRP = features_SRP[:, :75]
    features_NMF = features_NMF[:, :40]
    features_dec = np.hstack((features_SRP, features_PCA, features_NMF))

    # correlation matrix of features with highest importance with label vector
    corr = f_regression(features, label)

    # data frame of important features with label vector
    f_selection = pd.DataFrame({'importance': corr[1], 'feature': features.columns}).sort_values(by=['importance'],
                                                                                                 ascending=[False])

    # feature engineered for better results
    eng_features_ = features.iloc[:, :].progress_apply(aggregate_row, axis=1)

    # selecting top 800 features
    col_init = pd.DataFrame({'importance': corr[1], 'features': features.columns}).sort_values(
        by=['importance'], ascending=[False])[:800]['features'].values
    features = features[col_init]

    ## concatinating the new 10 features
    features = pd.concat([features, eng_features_], axis=1)

    # concatenating the data with decomposition components
    features_dec = pd.DataFrame(features_dec)
    features = pd.concat([features, features_dec], axis=1)

    # finding the feature importance using the same model and returning top 150 features
    model = LGBMRegressor(objective='regression', metric='rsme')
    model.fit(features, label)

    num_features = 150

    col_final = pd.DataFrame({'importance': model.feature_importances_,
                              'feature': features.columns}).sort_values(by=['importance'],
                                                                        ascending=[False])[:num_features][
        'feature'].values

    features = features[col_final]

    return (col_init, col_final, features, label)


# function to add new features
def aggregate_row(row):
    '''Function to add new features with non zero stats'''

    # if the values are non zero then add new features
    non_zero_values = row.iloc[np.array(row).nonzero()].astype(float)
    if non_zero_values.empty:

        aggs = {
            'non_zero_mean': 0.0,
            'non_zero_std': 0.0,
            'non_zero_max': 0.0,
            'non_zero_min': 0.0,
            'non_zero_sum': 0.0,
            'non_zero_skewness': 0.0,
            'non_zero_kurtosis': 0.0,
            'non_zero_moment': 0.0,
            'non_zero_log_q1': 0.0,
            'non_zero_log_q3': 0.0
        }
    else:
        aggs = {
            'non_zero_mean': non_zero_values.mean(),
            'non_zero_std': non_zero_values.std(),
            'non_zero_max': non_zero_values.max(),
            'non_zero_min': non_zero_values.min(),
            'non_zero_sum': non_zero_values.sum(),
            'non_zero_skewness': skew(non_zero_values),
            'non_zero_kurtosis': kurtosis(non_zero_values),
            'non_zero_moment': moment(non_zero_values),
            'non_zero_log_q1': np.percentile(np.log1p(non_zero_values), q=25),
            'non_zero_log_q3': np.percentile(np.log1p(non_zero_values), q=75)
        }
    return pd.Series(aggs, index=list(aggs.keys()))


# loading the dataset
data = pd.read_csv('train.csv')
# preprocessing
col_init, col_final, features, label = preprocessing(data)

# initializing the model
model = LGBMRegressor(objective='regression', metric='rmse',
                      bagging_fraction=0.6, bagging_freq=4,
                      boosting_type='gbdt', feature_fraction=0.91,
                      lambda_l1=0.45, lambda_l2=0.4,
                      learning_rate=0.01, max_bin=800,
                      max_depth=16, min_data_in_leaf=16,
                      n_estimators=250, num_iterations=435, num_leaves=65)

# fitting the model
model.fit(features, label)

# dumping the model
joblib.dump(model, 'best_model.pkl')
