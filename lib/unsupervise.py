import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import tensorflow.keras as keras
import tensorflow as tf
from sklearn.model_selection import ShuffleSplit


def kmeans_elbow(data, col_features, max_cluster, diff=False):
    K = range(1, max_cluster + 1)
    X = data[col_features]
    params = pd.DataFrame(columns=['distortion', 'inertia'], index=K)
    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        params.loc[k, 'distortion'] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[
            0]
        params.loc[k, 'inertia'] = kmeanModel.inertia_

    if diff:
        return params.diff()
    else:
        return params


def hierarchial_clustering(data, col_features, link='ward'):
    X = data[col_features]
    Z = linkage(X, 'ward')
    c, coph_dists = cophenet(Z, pdist(X))
    print(c)
    print(Z)
    print(X[col_features].shape)

    # calculate full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.show()

def auto_encoder(data, col_features, max_cluster=5):
    dim_size = len(col_features)
    sss = ShuffleSplit(n_splits=max_cluster, test_size=0.3, random_state=123)
    train_idx, test_idx = sss.split(data[col_features])

    for i in range(0, max_cluster):
        inputs = keras.Input(shape=(dim_size,))
        encoder_1 = keras.layers.Dense(dim_size)(inputs)
        encoder_2 = keras.layers.Dense(dim_size)(encoder_1)
        encoder_out = keras.layers.Dense(max_cluster, activation=tf.nn.softmax)(encoder_2)
        decoder_in = keras.layers.Dense(dim_size)(encoder_out)
        decoder_2 = keras.layers.Dense(dim_size)(decoder_in)

        auto_encoder = keras.Model(inputs=inputs, outputs=decoder_2, name='autoencoder')
        compressor = keras.Model(inputs, encoder_out)

        auto_encoder.compile(metrics=['mse'], loss=tf.losses.MeanSquaredError)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

        auto_encoder.fit(data[col_features].iloc[train_idx[i]],
                         data[col_features].iloc[train_idx[i]],
                         epochs=100,
                         callbacks=[callback],
                         validation_data=(data[col_features].iloc[test_idx[i]], data[col_features].iloc[test_idx[i]]))


