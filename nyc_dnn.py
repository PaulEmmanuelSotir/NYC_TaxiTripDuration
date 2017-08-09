#!/usr/bin/python
# -*- coding: utf-8 -*-
""" NYC Taxi Trip Duration - Kaggle competion
Note that the code could probably be greatly simplified using tf.train.Supervisor and tf.contrib.learn.dnnregressor,
but we prefer to define model by hand here to learn more about tensorflow python API.

TODO:
    * save best performing model instead of last one
    * test model without softmax regression but with batch norm. to compare rmse with softmax regression model
    * try residual blocks on deeper architecture

.. See https://github.com/PaulEmmanuelSotir/NYC_TaxiTripDuration
"""
import os
import math
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import utils

__all__ = ['load_data', 'build_model', 'train']

DEFAULT_HYPERPARAMETERS = {'lr': 0.005,
                           'lr_decay': 0.5,
                           'hidden_size': 512,
                           'batch_size': 4*1024,
                           'residual_blocks': 3,
                           'duration_std_margin': 6,
                           'dropout_keep_prob': 0.78,
                           'l2_regularization': 0.01,
                           'duration_resolution': 512,
                           'activation': tf.nn.tanh}

TEST_SIZE = 0.07
TRAINING_EPOCHS = 300
DISPLAY_STEP_PREDIOD = 2
ALLOW_GPU_MEM_GROWTH = True

def _xavier_init(fan_in, fan_out):
    return tf.random_normal([fan_in, fan_out], stddev=math.sqrt(3. / (fan_in + fan_out)))

def _softmax_to_duration(softmax, std, mean, duration_std_margin, duration_resolution):
    """ Inverse logistic function (logit function)
    This function is used to convert softmax output layer to a trip duration in a differentiable way so that we can perform softmax regression with L2 loss.
    Each softmax output probability weights a different trip duration value.
    These values are choosed to discretize trip duration so that area under gaussian curve is constant between those (follows a logit function).
    """
    max_x = tf.exp(duration_std_margin * std) / (1. + tf.exp(duration_std_margin * std))
    min_x = tf.exp(-duration_std_margin * std) / (1. + tf.exp(-duration_std_margin * std))
    mean_indice = tf.reduce_mean(tf.multiply(softmax, tf.range(0., duration_resolution, dtype=tf.float32)), axis=1) # TODO: make sure ignoring first softmax output (multiplied by 0) isn't a problem
    x = mean_indice * (max_x - min_x) + min_x
    pred = tf.log(x / (1 - x)) + mean
    return pred

def _haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Source: https://gis.stackexchange.com/a/56589/15183
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    meters = 1000 * 6367 * c
    return meters

def load_data(train_path, test_path):
    trainset = pd.read_csv(train_path)
    predset = pd.read_csv(test_path)

    # Parse and vectorize dates
    def _preprocess_date(dataset, field):
        dataset[field] = pd.to_datetime(dataset[field])
        dataset[field + '_hour'] = dataset[field].dt.hour
        dataset[field + '_min'] = dataset[field].dt.minute
        dataset[field + '_weekday'] = dataset[field].dt.weekday
        dataset[field + '_day'] = dataset[field].dt.day
    _preprocess_date(trainset, 'pickup_datetime')
    _preprocess_date(predset, 'pickup_datetime')
    # Vectorize flags
    trainset.store_and_fwd_flag, _ = pd.factorize(trainset.store_and_fwd_flag)
    predset.store_and_fwd_flag, _ = pd.factorize(predset.store_and_fwd_flag)
    # Add haversine distance feature
    trainset['distance'] = _haversine(trainset.pickup_longitude, trainset.pickup_latitude, trainset.dropoff_longitude, trainset.dropoff_latitude)
    predset['distance'] = _haversine(predset.pickup_longitude, predset.pickup_latitude, predset.dropoff_longitude, predset.dropoff_latitude)

    # Transform target trip durations to log(trip durations + 1) (permits to get a gaussian distribution of trip_durations, see data exploration notebook)
    targets = np.log(trainset.trip_duration + 1).values.reshape([-1, 1])
    # Get trip duration mean and std dev for duration softmax regression
    mean, std = np.mean(targets), np.std(targets)

    # Remove unused feature columns
    features = [key for key in trainset.keys().intersection(predset.keys()) if key != 'id' and key != 'pickup_datetime']
    data = trainset[features].get_values()
    # Split dataset into trainset and testset
    train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=TEST_SIZE)
    # Normalize feature columns
    standardizer = preprocessing.StandardScaler()
    train_data = standardizer.fit_transform(train_data)
    test_data = standardizer.transform(test_data)
    pred_data = standardizer.transform(predset[features].get_values())
    return features, (predset['id'], pred_data), (train_data, test_data, train_targets, test_targets), (std, mean)

def _dnn(X, weights, biases, dropout_keep_prob, activation):
    layers = [tf.nn.dropout(activation(tf.add(tf.matmul(X, weights[0]), biases[0])), dropout_keep_prob)]
    for w, b in zip(weights[1:-1], biases[1:-1]):
        # Fully connected residual block
        input = activation(tf.layers.batch_normalization(layers[-1]))
        dense1 = tf.nn.dropout(activation(tf.add(tf.matmul(input, w[0]), b[0])), dropout_keep_prob)
        dense1_bn = tf.layers.batch_normalization(dense1)
        dense2 = tf.nn.dropout(tf.add(tf.matmul(dense1_bn, w[1]), b[1]), dropout_keep_prob)
        layers.append(tf.add(dense2, layers[-1]))
    input = activation(tf.layers.batch_normalization(layers[-1]))
    return tf.add(tf.matmul(input, weights[-1]), biases[-1], name='logits')

def build_model(n_input, hp, target_std, target_mean):
    # Input placeholders
    X = tf.placeholder(tf.float32, [None, n_input], name='X')
    y = tf.placeholder(tf.float32, [None, 1], name='y')
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    with tf.variable_scope('dnn'):
        # Declare tensorflow constants (hyperparameters)
        target_std, target_mean = tf.constant(target_std, dtype=tf.float32), tf.constant(target_mean, dtype=tf.float32)
        hidden_size, resolution, std_margin = hp['hidden_size'], tf.constant(hp['duration_resolution'], dtype=tf.float32), tf.constant(hp['duration_std_margin'], dtype=tf.float32)
        # Define DNN weight and bias variables
        def _create_weights(shape, name):
            weights = tf.Variable(_xavier_init(*shape), name=name)
            utils.visualize_weights(weights, name=name)
            return weights
        with tf.variable_scope('input_layer'):
            weights = [_create_weights((n_input, hidden_size), name='weights')]
            biases = [tf.Variable(tf.random_normal([hidden_size]), name='b1')]
        for i in range(1, hp['residual_blocks']):
            with tf.variable_scope('residual_block' + str(i)):
                weights.append((_create_weights((hidden_size, hidden_size), name='weights_0'), _create_weights((hidden_size, hidden_size), name='weights_1')))
                biases.append((tf.Variable(tf.random_normal([hidden_size]), name='bias_0'), tf.Variable(tf.random_normal([hidden_size]), name='bias_1')))
        with tf.variable_scope('ouput_layer'):
            weights.append(_create_weights((hidden_size, hp['duration_resolution']), name='weights'))
            biases.append(tf.Variable(tf.random_normal([1]), name='bias'))
        # Build fully connected layers
        logits = _dnn(X, weights, biases, dropout_keep_prob, hp['activation'])

    # Define loss and optimizer
    pred = _softmax_to_duration(tf.nn.softmax(logits), target_std, target_mean, std_margin, resolution)
    l2_regularization = tf.reduce_sum([tf.nn.l2_loss(w) for w in weights])
    rmse = tf.sqrt(tf.losses.mean_squared_error(y, tf.reshape(pred, [-1, 1])))
    optimizer = tf.train.AdamOptimizer(learning_rate=hp['lr']).minimize(rmse + hp['l2_regularization'] * l2_regularization)
    # Variable initialization operation
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    return pred, rmse, optimizer, (X, y, dropout_keep_prob), tf.train.Saver(), init_op

def train(model, dataset, epochs, hp, save_dir=None, predset=None):
    # Unpack parameters
    train_data, test_data, train_targets, test_targets = dataset
    pred, rmse, optimizer, placeholders, saver, init_op = model
    X, y, dropout_keep_prob = placeholders

    # Add pred and rmse to summary
    tf.summary.histogram('pred', pred)
    rmse_summary_op = tf.summary.scalar('rmse', rmse)

    # Start tensorlfow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = ALLOW_GPU_MEM_GROWTH
    with tf.Session(config=config) as sess:
        sess.run(init_op)

        # Create summary utils
        if save_dir is not None:
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(save_dir, sess.graph)
            summary_test_writer = tf.summary.FileWriter(os.path.join(save_dir, 'test_rmse'), sess.graph)

        # Training loop
        best_rmse = float("inf")
        for epoch in range(1, epochs):
            hp['lr'] = hp['lr'] * hp['lr_decay'] if (epoch % 5) == 0 else hp['lr']
            batch_count = len(train_data) // hp['batch_size']
            for _ in range(batch_count):
                indices = np.random.randint(len(train_data), size=hp['batch_size'])
                batch_xs = train_data[indices, :]
                batch_ys = train_targets[indices, :]
                sess.run(optimizer, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob: hp['dropout_keep_prob']})
            # Display progress
            if epoch % DISPLAY_STEP_PREDIOD == 0:
                if save_dir is not None:
                    summary, last_loss = sess.run([summary_op, rmse], feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob: 1.})
                    summary_writer.add_summary(summary, epoch)
                    summary, test_rmse = sess.run([rmse_summary_op, rmse], feed_dict={X: test_data, y: test_targets, dropout_keep_prob: 1.})
                    summary_test_writer.add_summary(summary, epoch)
                else:
                    last_loss = sess.run(rmse, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob: 1.})
                    test_rmse = sess.run(rmse, feed_dict={X: test_data, y: test_targets, dropout_keep_prob: 1.})
                if best_rmse >= test_rmse:
                    best_rmse = test_rmse
                    if save_dir is not None:
                        print('Saving model...')
                        saver.save(sess, save_dir)
                print("Epoch=%03d/%03d, last_loss=%.6f, test_rmse=%.6f" % (epoch, epochs, last_loss, test_rmse))
        print("Training done, best_rmse=%.6f" % best_rmse)

        # Restore best model and apply prediction
        if save_dir is not None:
            saver.restore(sess, save_dir)
        if predset is not None:
            print('Applying trained model on prediction set')
            predictions = []
            for batch in np.array_split(predset, len(predset) // hp['batch_size']):
                predictions.extend(sess.run(pred, feed_dict={X: batch, dropout_keep_prob: 1.}).flatten())
            return test_rmse, predictions
        return test_rmse

def main():
    # Parse cmd arguments
    parser = argparse.ArgumentParser(description='Trains NYC Taxi trip duration fully connected neural network model for Kaggle competition submission.')
    parser.add_argument('--floyd-job', action='store_true', help='Change working directories for training on Floyd service')
    args = parser.parse_args()
    save_dir = '/output/model/' if args.floyd_job else './model/'
    train_set_path = '/input/train.csv' if args.floyd_job else './NYC_taxi_data_2016/train.csv'
    pred_set_path = '/input/test.csv' if args.floyd_job else './NYC_taxi_data_2016/test.csv'

    # Parse and preprocess data
    features, (pred_ids, predset), dataset, (target_std, target_mean) = load_data(train_set_path, pred_set_path)

    # Build model
    hyperparameters = DEFAULT_HYPERPARAMETERS
    model = build_model(len(features), hyperparameters, target_std, target_mean)
    
    # Add trainable variables to summary
    for v in tf.trainable_variables():
        tf.summary.histogram(v.name, v)

    # Train model
    print('Model built, starting training.')
    test_rmse, predictions = train(model, dataset, TRAINING_EPOCHS, hyperparameters, save_dir, predset)

    # Save predictions to csv file for Kaggle submission
    predictions = np.int32(np.round(np.exp(predictions))) - 1
    pd.DataFrame(np.column_stack([pred_ids, predictions]), columns=['id', 'trip_duration']).to_csv(os.path.join(save_dir, 'preds.csv'), index=False)

if __name__ == '__main__':
    main()

"""
TODO: try cross entropy loss...
def _discretize_duration(y, std, mean, duration_std_margin, duration_resolution):
    min_x = tf.exp(duration_std_margin * std) / (1. + tf.exp(duration_std_margin * std))
    max_x = tf.exp(-duration_std_margin * std) / (1. + tf.exp(-duration_std_margin * std))
    vals = tf.exp(tf.reshape(y, [-1]) - mean)
    indices = ((min_x - vals / (1. + vals)) / (max_x - min_x) + 1.) * duration_resolution
    indices = tf.clip_by_value(indices, 0., duration_resolution)
    return tf.cast(tf.round(indices), tf.int32)
"""
