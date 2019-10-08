import pandas as pd
from sklearn.metrics import auc
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

def predictor_power(data, col_features, col_target):

    if isinstance(col_target, list):
        df_power = pd.DataFrame(columns=col_target, index=col_features)
        for i in range(0, len(col_target)):
            for j in range(0, len(col_features)):
                data = data.sort_values(by=[col_features[j]])
                gini = 2*(auc(data[col_features[j]], data[col_target[i]])) - 1
                print(f'{col_features[j]}: {gini}')
                df_power.loc[col_features[j], col_target[i]] = gini
    else:
        df_power = pd.DataFrame(index=col_features)
        for j in range(0, len(col_features)):
            data = data.sort_values(by=[col_features[j]])
            gini = 2 * (auc(data[col_features[j]], data[col_target])) - 1
            # print(f'{col_features[j]}: {gini}')
            df_power.loc[col_features[j], col_target] = gini
    return df_power

def kmeans_elbow(data, col_features, max_cluster, diff=False):
    K = range(1, max_cluster+1)
    X = data[col_features]
    params = pd.DataFrame(columns=['distortion', 'inertia'], index=K)
    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        params.loc[k, 'distortion'] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]
        params.loc[k, 'inertia'] = kmeanModel.inertia_

    if diff: return params.diff()
    else: return params