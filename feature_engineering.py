#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Most of this code is from Nir Malbin's notebook: https://www.kaggle.com/donniedarko/darktaxi-tripdurationprediction-lb-0-385 """

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


def _geohash(features, prefix, longitude, latitude, precision):
    """ Encode a (lon, lat) pair to a GeoHash.
        Code inspired from https://github.com/transitland/mapzen-geohash/blob/master/mzgeohash/geohash.py
    """

    def _float_to_bits(value, lower, upper):
        """ Convert a float to a list of GeoHash bits """
        middle = 0.0
        for bit in range(int(precision / 2)):
            fname = prefix + str(bit)
            if fname not in features:
                features[fname] = []
            byte = 0
            for bit in range(5):
                if value >= middle:
                    lower = middle
                    byte += 2**bit
                else:
                    upper = middle
                middle = (upper + lower) / 2
            features[fname].append(byte)

    # Half the length for each component.
    _float_to_bits(longitude, lower=-180.0, upper=180.0)
    _float_to_bits(latitude, lower=-90.0, upper=90.0)


def _clustering(X, df_all):
    kmeans = MiniBatchKMeans(n_clusters=8**2, batch_size=32**3).fit(X)
    df_all['pickup_cluster'] = kmeans.predict(df_all[['pickup_latitude', 'pickup_longitude']])
    df_all['dropoff_cluster'] = kmeans.predict(df_all[['dropoff_latitude', 'dropoff_longitude']])
    df_all['pickup_datetime_group'] = df_all['pickup_datetime'].dt.round('60min')


def load_features(trainset, testset):
    features = ['vendor_id', 'passenger_count', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'pickup_pca0',
                'pickup_pca1', 'dropoff_pca0', 'dropoff_pca1', 'pca_manhattan', 'month', 'weekofyear', 'weekday', 'hour', 'week_delta', 'week_delta_sin', 'hour_sin']
    df_all = pd.concat((pd.read_csv(trainset), pd.read_csv(testset)))
    df_all['pickup_datetime'] = df_all['pickup_datetime'].apply(pd.Timestamp)
    df_all['dropoff_datetime'] = df_all['dropoff_datetime'].apply(pd.Timestamp)
    df_all['trip_duration_log'] = np.log(df_all['trip_duration'] + 1)

    # Remove abnormal locations
    X = np.vstack((df_all[['pickup_latitude', 'pickup_longitude']], df_all[['dropoff_latitude', 'dropoff_longitude']]))
    min_lat, min_lng = X.mean(axis=0) - X.std(axis=0)
    max_lat, max_lng = X.mean(axis=0) + X.std(axis=0)
    X = X[(X[:, 0] > min_lat) & (X[:, 0] < max_lat) & (X[:, 1] > min_lng) & (X[:, 1] < max_lng)]

    # Get PCA features on location
    pca = PCA().fit(X)
    df_all['pickup_pca0'] = pca.transform(df_all[['pickup_latitude', 'pickup_longitude']])[:, 0]
    df_all['pickup_pca1'] = pca.transform(df_all[['pickup_latitude', 'pickup_longitude']])[:, 1]
    df_all['dropoff_pca0'] = pca.transform(df_all[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    df_all['dropoff_pca1'] = pca.transform(df_all[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
    df_all['pca_manhattan'] = (df_all['dropoff_pca0'] - df_all['pickup_pca0']).abs() + (df_all['dropoff_pca1'] - df_all['pickup_pca1']).abs()

    # Geohash
    precision = 12
    geohash_features = {}
    for _, row in df_all.iterrows():
        _geohash(geohash_features, 'pickup_geohash_', row['pickup_latitude'], row['pickup_longitude'], precision=precision)
        _geohash(geohash_features, 'dropoff_geohash_', row['dropoff_latitude'], row['dropoff_longitude'], precision=precision)
    df_all = df_all.join(pd.DataFrame(geohash_features))
    features.extend(list(geohash_features.keys()))

    # Date times
    df_all['month'] = df_all['pickup_datetime'].dt.month
    df_all['weekofyear'] = df_all['pickup_datetime'].dt.weekofyear
    df_all['weekday'] = df_all['pickup_datetime'].dt.weekday
    df_all['hour'] = df_all['pickup_datetime'].dt.hour
    df_all['week_delta'] = df_all['pickup_datetime'].dt.weekday + ((df_all['pickup_datetime'].dt.hour + (df_all['pickup_datetime'].dt.minute / 60.0)) / 24.0)

    # Make time features cyclic
    df_all['week_delta_sin'] = np.sin((df_all['week_delta'] / 7) * np.pi)**2
    df_all['hour_sin'] = np.sin((df_all['hour'] / 24) * np.pi)**2

    # Traffic (Count trips over 60min)
    df_counts = df_all.set_index('pickup_datetime')[['id']].sort_index()
    df_all = df_all.merge(df_counts, on='id', how='left')

    # K means clustering
    _clustering(X, df_all)

    # Get test set and train set from df_all
    X_train = df_all[df_all['trip_duration'].notnull()][features].values
    y_train = df_all[df_all['trip_duration'].notnull()]['trip_duration_log'].values.flatten()
    X_test = df_all[df_all['trip_duration'].isnull()][features].values
    test_ids = df_all[df_all['trip_duration'].isnull()]['id'].values

    return X_train, X_test, y_train, test_ids
