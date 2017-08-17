#!/usr/bin/python
# -*- coding: utf-8 -*-
""" NYC Taxi Trip Duration - Kaggle competion
Note that the code could probably be greatly simplified using tf.train.Supervisor and tf.contrib.learn.dnnregressor,
but we prefer to define model by hand here to learn more about tensorflow python API (and for more flexibility).

TODO:
    * use CV
    * approximate hessian with a meta model and perform newton method to update parameters at each step (reduces the need of hyperparameters like learning rate, ... and may converge faster)
    * try cross entropy instead of rmsle

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

DEFAULT_HYPERPARAMETERS = {'epochs': 300,
                           'lr': 0.0005,
                           'opt': {'algo': tf.train.AdamOptimizer},
                           'depth': 4,
                           'hidden_size': 512,
                           'batch_size': 1024,
                           'early_stopping': 30,
                           'max_norm_threshold': 1.,
                           'duration_std_margin': 6,
                           'dropout_keep_prob': 0.83,
                           'output_size': 256}

TEST_SIZE = 0.07
TRAINING_EPOCHS = 300
DISPLAY_STEP_PREDIOD = 2
ALLOW_GPU_MEM_GROWTH = True

def _xavier_init(fan_in, fan_out):
    return tf.random_normal([fan_in, fan_out], stddev=math.sqrt(3. / (fan_in + fan_out)))

def _max_norm_regularizer(threshold, collection):
    def _max_norm(weights):
        # Apply max-norm regularization on weights matrix columns
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=1)
        clip_weights = tf.assign(weights, clipped, name='max_norm')
        tf.add_to_collection(collection, clip_weights)
    return _max_norm

def _softmax_to_duration(softmax, std, mean, duration_std_margin, output_size):
    """ Inverse logistic function (logit function)
    This function is used to convert softmax output layer to a trip duration in a differentiable way so that we can perform softmax regression with L2 loss.
    Each softmax output probability weights a different trip duration value.
    These values are choosed to discretize trip duration so that area under gaussian curve is constant between those (follows a logit function).
    """
    max_x = tf.exp(duration_std_margin * std) / (1. + tf.exp(duration_std_margin * std))
    min_x = tf.exp(-duration_std_margin * std) / (1. + tf.exp(-duration_std_margin * std))
    mean_indice = tf.reduce_mean(tf.multiply(softmax, tf.range(0., output_size, dtype=tf.float32)), axis=1)
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

def load_data(dataset_dir, train_file, test_file, knn_train_file, knn_test_file, knn_pred_file):
    trainset = pd.read_csv(os.path.join(dataset_dir, train_file))
    predset = pd.read_csv(os.path.join(dataset_dir, test_file))

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

    return pred_data.shape[1], (predset['id'], pred_data), (train_data, test_data, train_targets, test_targets), (std, mean)
def _layer(x, shape, dropout_keep_prob, name, batch_norm=True, summarize=True, activation=tf.nn.tanh, weights_regularizer=None):
    with tf.variable_scope(name):
        weights = tf.Variable(_xavier_init(*shape), name='w')
        if weights_regularizer is not None:
            weights_regularizer(weights)
        bias = tf.Variable(tf.random_normal([shape[1]]), name='b')
        print(weights.shape)
        print(x.shape)
        logits = tf.add(tf.matmul(x, weights), bias)
        dense = activation(logits) if activation is not None else logits
        dense_bn = tf.layers.batch_normalization(dense) if batch_norm else dense
        dense_do = dense_bn if dropout_keep_prob == 1. else tf.nn.dropout(dense_bn, dropout_keep_prob)
        if summarize:
            utils.visualize_weights(weights, name='weights')
            tf.summary.histogram('bias', bias)
    return dense_do

def build_model(n_input, hp, target_std, target_mean, summarize_parameters=True):
    # Input placeholders
    X = tf.placeholder(tf.float32, [None, n_input], name='X')
    y = tf.placeholder(tf.float32, [None, 1], name='y')
    lr = tf.placeholder(tf.float32, name='lr')
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    with tf.variable_scope('dnn'):
        # Declare tensorflow constants (hyperparameters)
        target_std, target_mean = tf.constant(target_std, dtype=tf.float32), tf.constant(target_mean, dtype=tf.float32)
        hidden_size, resolution, std_margin = hp['hidden_size'], tf.constant(hp['output_size'], dtype=tf.float32), tf.constant(hp['duration_std_margin'], dtype=tf.float32)
        # Define DNN layers
        wreg = _max_norm_regularizer(hp['max_norm_threshold'], 'max_norm')
        layer = _layer(X, (n_input, hidden_size), dropout_keep_prob, 'input_layer', batch_norm=False, summarize=summarize_parameters, weights_regularizer=wreg)
        for i in range(1, hp['depth'] - 1):
            layer = _layer(layer, (hidden_size, hidden_size), dropout_keep_prob, 'layer_' + str(i), summarize=summarize_parameters, weights_regularizer=wreg)
        logits = _layer(layer, (hidden_size, hp['output_size']), 1., 'output_layer', summarize=summarize_parameters, activation=None, weights_regularizer=wreg)

    # Define loss and optimizer
    normalized_pred = logits if hp['output_size'] == 1 else _softmax_to_duration(tf.nn.softmax(logits), target_std, target_mean, std_margin, resolution)
    normalized_rmse = tf.sqrt(tf.losses.mean_squared_error((y - target_mean) / target_std, tf.reshape(normalized_pred, [-1, 1])))
    pred = normalized_pred * target_std + target_mean
    rmse = tf.sqrt(tf.losses.mean_squared_error(y, tf.reshape(pred, [-1, 1])))
    opt_algorithm = hp['opt']['algo']
    optimizer = opt_algorithm(learning_rate=lr) if opt_algorithm is not tf.train.MomentumOptimizer else opt_algorithm(learning_rate=lr, momentum=hp['opt']['m'])
    grads_and_vars = optimizer.compute_gradients(normalized_rmse)
    # Add pred, rmse and gradients to submmary
    with tf.name_scope('dnn_gradients'):
        for g, v in grads_and_vars:
            if g is not None and summarize_parameters:
                tf.summary.histogram(v.name, g)
    optimize = optimizer.apply_gradients(grads_and_vars)
    tf.summary.histogram('pred', pred)
    tf.summary.scalar('rmse', rmse)
    # Variable initialization operation
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    return pred, rmse, optimize, (X, y, lr, dropout_keep_prob), tf.train.Saver(), init_op

def train(model, dataset, hp, save_dir=None, predset=None):
    # Unpack parameters
    train_data, test_data, train_targets, test_targets = dataset
    pred, rmse, optimizer, placeholders, saver, init_op = model
    X, y, lr, dropout_keep_prob = placeholders

    # Start tensorflow session
    with tf.Session(config=utils.tf_config(ALLOW_GPU_MEM_GROWTH)) as sess:
        sess.run(init_op)

        # Create summary utils
        if save_dir is not None:
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(save_dir, sess.graph)

        # Get max-norm regularization operations
        max_norm_ops = tf.get_collection("max_norm")

        # Training loop
        best_rmse = float("inf")
        steps_since_improvement = 0
        for epoch in range(1, hp['epochs']):
            # Apply learning rate decay every LR_DECAY_PERIOD epochs
            if (epoch % LR_DECAY_PERIOD) == 0  and (hp['opt'] is tf.train.MomentumOptimizer or hp['opt'] is tf.train.GradientDescentOptimizer):
                hp['lr'] *= hp['opt']['lr_decay']
            # Train model using minibatches
            batch_count = len(train_data) // hp['batch_size']
            for _ in range(batch_count):
                indices = np.random.randint(len(train_data), size=hp['batch_size'])
                batch_xs = train_data[indices, :]
                batch_ys = train_targets[indices, :]
                sess.run(optimizer, feed_dict={X: batch_xs, y: batch_ys, lr: hp['lr'], dropout_keep_prob: hp['dropout_keep_prob']})
                # Apply max norm regularization
                sess.run(max_norm_ops)
            # Evaluate model and display progress
            steps_since_improvement += 1
            if epoch % DISPLAY_STEP_PREDIOD == 0:
                if save_dir is not None:
                    summary, test_rmse = sess.run([summary_op, rmse], feed_dict={X: test_data, y: test_targets, dropout_keep_prob: 1.})
                    summary_writer.add_summary(summary, epoch)
                else:
                    test_rmse = sess.run(rmse, feed_dict={X: test_data, y: test_targets, dropout_keep_prob: 1.})
                if best_rmse > test_rmse:
                    steps_since_improvement = 0
                    best_rmse = test_rmse
                    if save_dir is not None:
                        print('Saving model...')
                        saver.save(sess, save_dir)
                print("Epoch=%03d/%03d, test_rmse=%.6f" % (epoch, hp['epochs'], test_rmse))
                if steps_since_improvement >= hp['early_stopping']:
                    print('Early stopping.')
                    break
        print("Training done, best_rmse=%.6f" % best_rmse)

        # Restore best model and apply prediction
        if save_dir is not None:
            saver.restore(sess, save_dir)
        if predset is not None:
            print('Applying best trained model on prediction set')
            predictions = []
            for batch in np.array_split(predset, len(predset) // hp['batch_size']):
                predictions.extend(sess.run(pred, feed_dict={X: batch, dropout_keep_prob: 1.}).flatten())
            return best_rmse, predictions
        return best_rmse

def main(_=None):
    # Define working directories
    save_dir = '/output/models/' if tf.flags.FLAGS.floyd_job else './models/'
    dataset_dir = '/input/' if tf.flags.FLAGS.floyd_job else './NYC_taxi_data_2016/'
    
    # Set log level to debug
    tf.logging.set_verbosity(tf.logging.INFO)

    # Parse and preprocess data
    knn_files = ('knn_train_features.npz', 'knn_test_features.npz', 'knn_pred_features.npz')
    features_len, (pred_ids, predset), dataset, (target_std, target_mean) = load_data(dataset_dir, 'train.csv', 'test.csv', *knn_files)

    # Build model
    hyperparameters = DEFAULT_HYPERPARAMETERS
    model = build_model(features_len, hyperparameters, target_std, target_mean)

    # Train model
    print('Model built, starting training.')
    test_rmse, predictions = train(model, dataset, hyperparameters, save_dir, predset)

    # Save predictions to csv file for Kaggle submission
    predictions = np.int32(np.round(np.exp(predictions))) - 1
    pd.DataFrame(np.column_stack([pred_ids, predictions]), columns=['id', 'trip_duration']).to_csv(os.path.join(save_dir, 'preds.csv'), index=False)

if __name__ == '__main__':
    tf.flags.DEFINE_bool('floyd-job', False, 'Change working directories for training on Floyd.')
    tf.app.run()
