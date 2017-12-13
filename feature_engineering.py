#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Load dataset and preprocess data before model training """

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
import shutil
import os

from joblib import Memory

__all__ = ['load_data']


def _osrm(datadir):
    features = ['total_distance', 'total_travel_time', 'number_of_steps']
    fr1 = pd.read_csv(os.path.join(datadir, 'osrm/fastest_routes_train_part_1.csv'), usecols=['id', *features])
    fr2 = pd.read_csv(os.path.join(datadir, 'osrm/fastest_routes_train_part_2.csv'), usecols=['id', *features])
    test_street_info = pd.read_csv(os.path.join(datadir, 'osrm/fastest_routes_test.csv'), usecols=['id', *features])
    train_street_info = pd.concat((fr1, fr2))
    return train_street_info, test_street_info, features


def _bucketize(train_targets, valid_targets, bucket_count):
    """ Process buckets from train targets and deduce labels of trainset and testset """
    sorted_targets = np.sort(train_targets)
    bucket_size = len(sorted_targets) // bucket_count
    buckets = [sorted_targets[i * bucket_size: (1 + i) * bucket_size] for i in range(bucket_count)]
    bucket_maxs = [np.max(b) for b in buckets]
    bucket_maxs[-1] = float('inf')

    # Bucketize targets (labels are bucket indices)
    def _find_indice(value): return np.searchsorted(bucket_maxs, value)
    train_labels = np.vectorize(_find_indice)(train_targets)
    valid_labels = np.vectorize(_find_indice)(valid_targets)
    # Process buckets means
    buckets_means = [np.mean(bucket) for bucket in buckets]
    return train_labels, valid_labels, buckets_means


def load_data(datadir, trainset, testset, valid_size, output_size, embed_discrete_features=False, max_distinct_values=None, cache_read_only=False):
    if cache_read_only:
        dest = '/output/cache'
        shutil.copytree(datadir, dest)
        datadir = dest
    memory = Memory(cachedir=os.path.join(datadir, 'cache'))

    @memory.cache(ignore=['datadir'])
    def _cached_light_load_data(datadir, trainset, testset, valid_size, embed_discrete_features, max_distinct_values):
        features = ['vendor_id', 'passenger_count', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        df_all = pd.concat((pd.read_csv(os.path.join(datadir, trainset)), pd.read_csv(os.path.join(datadir, testset))))
        df_all['pickup_datetime'] = df_all['pickup_datetime'].apply(pd.Timestamp)
        df_all['dropoff_datetime'] = df_all['dropoff_datetime'].apply(pd.Timestamp)
        df_all['trip_duration_log'] = np.log(df_all['trip_duration'] + 1)

        def _add_feature(name, value):
            features.append(name)
            df_all[name] = value

        _, indices = np.unique(df_all['id'], return_inverse=True)
        _add_feature('id_idx', indices)
        # Date time features
        pickup_dt = df_all['pickup_datetime'].dt
        _add_feature('week_delta', pickup_dt.weekday + ((pickup_dt.hour + (pickup_dt.minute / 60.0)) / 24.0))
        _add_feature('weekofyear', pickup_dt.weekofyear)
        _add_feature('weekday', pickup_dt.weekday)
        _add_feature('seconds', pickup_dt.second + pickup_dt.minute * 60.)
        _add_feature('month', pickup_dt.month)
        _add_feature('hour', pickup_dt.hour)

        # Get test set and train set from df_all
        X_train = df_all[df_all['trip_duration'].notnull()]
        y_train = df_all[df_all['trip_duration'].notnull()]['trip_duration_log'].values.flatten()
        X_test = df_all[df_all['trip_duration'].isnull()]
        test_ids = df_all[df_all['trip_duration'].isnull()]['id'].values

        # Add OSRM data
        train_street_info, test_street_info, osrm_features = _osrm(datadir)
        features.extend(osrm_features)
        X_train = X_train.merge(train_street_info, how='left', on='id')[features]
        X_test = X_test.merge(test_street_info, how='left', on='id')[features]

        # Fill missing osrm data
        mean_distance, mean_travel_time, mean_steps = X_train.total_distance.mean(), X_train.total_travel_time.mean(), X_train.number_of_steps.mean()

        def _fillnan(df):
            df.total_distance = df.total_distance.fillna(mean_distance)
            df.total_travel_time = df.total_travel_time.fillna(mean_travel_time)
            df.number_of_steps = df.number_of_steps.fillna(round(mean_steps))
        _fillnan(X_train)
        _fillnan(X_test)

        # Split dataset into trainset and testset
        train_data, valid_data, train_targets, valid_targets = train_test_split(X_train.values, y_train, test_size=valid_size, random_state=459)

        # Normalize feature columns
        standardizer = preprocessing.StandardScaler()
        train_data = standardizer.fit_transform(train_data)
        valid_data = standardizer.transform(valid_data)
        test_data = standardizer.transform(X_test.values)

        # Encode discrete features with less than 'max_distinct_values' unique values
        discrete_features = []
        if embed_discrete_features:
            for feature_idx in range(len(features)):
                train_feature = train_data[:, feature_idx]
                # TODO: apply a threshold on values count in order to filter out outlier values
                # (values, counts = np.unique(train_data[:, feature_idx], return_counts=True))
                unique_values = np.unique(train_feature)
                if max_distinct_values is None or len(unique_values) < max_distinct_values:
                    discrete_features.append((feature_idx, np.append(unique_values, [len(unique_values)])))

                    def _encode(array, encoded_values):
                        rslt = np.empty_like(array)
                        for val in np.unique(array):
                            encode_idx = np.argwhere(encoded_values == val)
                            arr_idx = np.argwhere(array == val)
                            rslt[arr_idx] = len(encoded_values) if len(encode_idx) == 0 else encode_idx[0, 0]
                        return rslt
                    unique_values, train_data[:, feature_idx] = np.unique(train_feature, return_inverse=True)
                    valid_data[:, feature_idx] = _encode(valid_data[:, feature_idx], unique_values)
                    test_data[:, feature_idx] = _encode(test_data[:, feature_idx], unique_values)

        return len(features), discrete_features, (test_ids, test_data), (train_data, valid_data, train_targets, valid_targets)

    # Parse and preprocess data
    features_len, discrete_features, (test_ids, testset), dataset = _cached_light_load_data(
        datadir, trainset, testset, valid_size, embed_discrete_features, max_distinct_values)
    (train_data, valid_data, train_targets, valid_targets) = dataset

    # Get buckets from train targets
    train_labels, valid_labels, bucket_means = _bucketize(train_targets, valid_targets, output_size)

    return features_len, discrete_features, (test_ids, testset), (train_data, valid_data, train_targets, valid_targets, train_labels, valid_labels), bucket_means
