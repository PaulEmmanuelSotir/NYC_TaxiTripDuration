#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Code from Nir Malbin's notebook: https://www.kaggle.com/donniedarko/darktaxi-tripdurationprediction-lb-0-385 """

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def arrays_bearing(lats1, lngs1, lats2, lngs2, R=6371):
    lats1_rads = np.radians(lats1)
    lats2_rads = np.radians(lats2)
    lngs_delta_rads = np.radians(lngs2 - lngs1)
    y = np.sin(lngs_delta_rads) * np.cos(lats2_rads)
    x = np.cos(lats1_rads) * np.sin(lats2_rads) - np.sin(lats1_rads) * np.cos(lats2_rads) * np.cos(lngs_delta_rads)
    return np.degrees(np.arctan2(y, x))

def clustering(X, df_all):
    kmeans = MiniBatchKMeans(n_clusters=8**2, batch_size=32**3).fit(X)

    df_all['pickup_cluster'] = kmeans.predict(df_all[['pickup_latitude', 'pickup_longitude']])
    df_all['dropoff_cluster'] = kmeans.predict(df_all[['dropoff_latitude', 'dropoff_longitude']])

    # Count how many trips are going to each cluster over time
    group_freq = '60min'
    df_dropoff_counts = df_all \
        .set_index('pickup_datetime') \
        .groupby([pd.TimeGrouper(group_freq), 'dropoff_cluster']) \
        .agg({'id': 'count'}) \
        .reset_index().set_index('pickup_datetime') \
        .groupby('dropoff_cluster').rolling('240min').mean() \
        .drop('dropoff_cluster', axis=1) \
        .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \
        .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'dropoff_cluster_count'})

    df_all['pickup_datetime_group'] = df_all['pickup_datetime'].dt.round(group_freq)
    df_all['dropoff_cluster_count'] = df_all[['pickup_datetime_group', 'dropoff_cluster']].merge(df_dropoff_counts, on=['pickup_datetime_group', 'dropoff_cluster'], how='left')['dropoff_cluster_count'].fillna(0)

def load_features(trainset='./NYC_taxi_data_2016/train.csv', testset='./NYC_taxi_data_2016/test.csv'):
    df_all = pd.concat((pd.read_csv(trainset), pd.read_csv(testset)))
    df_all['pickup_datetime'] = df_all['pickup_datetime'].apply(pd.Timestamp)
    df_all['dropoff_datetime'] = df_all['dropoff_datetime'].apply(pd.Timestamp)
    df_all['trip_duration_log'] = df_all['trip_duration'].apply(np.log)

    # Remove abnormal locations
    X = np.vstack((df_all[['pickup_latitude', 'pickup_longitude']], df_all[['dropoff_latitude', 'dropoff_longitude']]))
    min_lat, min_lng = X.mean(axis=0) - X.std(axis=0)
    max_lat, max_lng = X.mean(axis=0) + X.std(axis=0)
    X = X[(X[:,0] > min_lat) & (X[:,0] < max_lat) & (X[:,1] > min_lng) & (X[:,1] < max_lng)]

    # Get PCA features on location
    pca = PCA().fit(X)
    df_all['pickup_pca0'] = pca.transform(df_all[['pickup_latitude', 'pickup_longitude']])[:,0]
    df_all['pickup_pca1'] = pca.transform(df_all[['pickup_latitude', 'pickup_longitude']])[:,1]
    df_all['dropoff_pca0'] = pca.transform(df_all[['dropoff_latitude', 'dropoff_longitude']])[:,0]
    df_all['dropoff_pca1'] = pca.transform(df_all[['dropoff_latitude', 'dropoff_longitude']])[:,1]

    df_all['pca_manhattan'] = (df_all['dropoff_pca0'] - df_all['pickup_pca0']).abs() + (df_all['dropoff_pca1'] - df_all['pickup_pca1']).abs()

    # Bearing
    df_all['bearing'] = arrays_bearing(df_all['pickup_latitude'], df_all['pickup_longitude'], df_all['dropoff_latitude'], df_all['dropoff_longitude'])

    # Date times
    df_all['pickup_time_delta'] = (df_all['pickup_datetime'] - df_all['pickup_datetime'].min()).dt.total_seconds()
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
    df_counts['count_60min'] = df_counts.isnull().rolling('60min').count()['id']
    df_all = df_all.merge(df_counts, on='id', how='left')

    # K means clustering
    clustering(X, df_all)

    # Get test set and train set from df_all
    features = ['vendor_id', 'passenger_count', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',
                'pickup_pca0', 'pickup_pca1', 'dropoff_pca0', 'dropoff_pca1', 'bearing', 'pca_manhattan', 'pickup_time_delta',
                'month', 'weekofyear', 'weekday', 'hour', 'week_delta', 'week_delta_sin', 'hour_sin', 'count_60min', 'dropoff_cluster_count']
    X_train = df_all[df_all['trip_duration'].notnull()][features].values
    y_train = df_all[df_all['trip_duration'].notnull()]['trip_duration_log'].values
    X_test = df_all[df_all['trip_duration'].isnull()][features].values
    test_ids = df_all[df_all['trip_duration'].isnull()]['id'].values
    return X_train, X_test, y_train, test_ids
