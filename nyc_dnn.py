#!/usr/bin/python
# -*- coding: utf-8 -*-
""" NYC Taxi Trip Duration - Kaggle competion
Note that the code could probably be greatly simplified using tf.train.Supervisor and tf.contrib.learn.dnnregressor,
but we prefer to define model by hand here to learn more about tensorflow python API (and for more flexibility).

TODO:
    * use CV
    * summarize rmse and distributions/images separately at deifferent frequencies to reduce storage costs
    * try geohash for feature engineering

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

__all__ = ['load_data', 'get_buckets', 'build_model', 'train']

DEFAULT_HYPERPARAMETERS = {'epochs': 1200,
                           'lr': 0.0005,
                           'opt': {'algo': tf.train.AdamOptimizer},
                           'depth': 8,
                           'hidden_size': 512,
                           'batch_size': 1024,
                           'early_stopping': 70,
#                          'max_norm_threshold': 1.,
                           'duration_std_margin': 6,
                           'dropout_keep_prob': 0.83,
                           'output_size': 512}

TEST_SIZE = 100000
LR_DECAY_PERIOD = 5
DISPLAY_STEP_PREDIOD = 1
ALLOW_GPU_MEM_GROWTH = True

def get_buckets(train_targets, test_targets, bucket_count):
    # Process buckets from train targets
    bucket_size = len(train_targets) // bucket_count
    buckets = [train_targets[i * bucket_size: (1 + i) * bucket_size] for i in range(bucket_count)]
    # Bucketize targets (TODO: try soft classes)
    bucket_maxs = [np.max(b) for b in buckets]
    bucket_maxs[-1] = float('inf')
    find_indice = lambda value: np.searchsorted(bucket_maxs, value)
    train_labels = np.vectorize(find_indice)(train_targets)
    test_labels = np.vectorize(find_indice)(test_targets)
    # Process buckets means
    buckets_means = tf.constant([np.mean(bucket) for bucket in buckets], dtype=tf.float32, name='buckets_means')
    return train_labels, test_labels, buckets_means

def _buckets_to_duration(logits, bucket_means):
    return tf.reduce_sum(bucket_means * tf.nn.softmax(logits), axis=1)

def _max_norm_regularizer(threshold, collection):
    if threshold is not None:
        def _max_norm(weights):
            # Apply max-norm regularization on weights matrix columns
            clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=1)
            clip_weights = tf.assign(weights, clipped, name='max_norm')
            tf.add_to_collection(collection, clip_weights)
        return _max_norm
    return None

def load_data(dataset_dir, train_file, test_file, knn_train_file, knn_test_file, knn_pred_file):
    from feature_engineering import load_features
    data, pred_data, targets, pred_ids = load_features(os.path.join(dataset_dir, train_file), os.path.join(dataset_dir, test_file))

    # Split dataset into trainset and testset
    train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=TEST_SIZE)

    # Normalize feature columns
    standardizer = preprocessing.StandardScaler()
    train_data = standardizer.fit_transform(train_data)
    test_data = standardizer.transform(test_data)
    pred_data = standardizer.transform(pred_data)
    
    # Add KNN features
    knn_train_features = np.load(os.path.join(dataset_dir, knn_train_file))
    train_data = np.column_stack([train_data, knn_train_features / 4.])
    knn_test_features = np.load(os.path.join(dataset_dir, knn_test_file))
    test_data = np.column_stack([test_data, knn_test_features / 4.])
    knn_pred_features = np.load(os.path.join(dataset_dir, knn_pred_file))
    pred_data = np.column_stack([pred_data, knn_pred_features / 4.])

    return pred_data.shape[1], (pred_ids, pred_data), (train_data, test_data, train_targets, test_targets)

def _dense_layer(x, shape, dropout_keep_prob, name, batch_norm=True, summarize=True, activation=tf.nn.tanh, weights_regularizer=None):
    with tf.variable_scope(name):
        weights = tf.Variable(utils.xavier_init(*shape), name='w')
        if weights_regularizer is not None:
            weights_regularizer(weights)
        bias = tf.Variable(tf.random_normal([shape[1]]) if shape[1] > 1 else 0., name='b')
        logits = tf.add(tf.matmul(x, weights), bias)
        dense = activation(logits) if activation is not None else logits
        dense_bn = tf.layers.batch_normalization(dense) if batch_norm else dense
        dense_do = dense_bn if dropout_keep_prob == 1. else tf.nn.dropout(dense_bn, dropout_keep_prob)
        if summarize:
            image = tf.reshape(weights, [1, weights.shape[0].value, weights.shape[1].value, 1])
            tf.summary.image('weights', image)
            tf.summary.histogram('bias', bias)
    return dense_do

def build_model(n_input, hp, bucket_means, summarize_parameters=True):
    # Input placeholders
    X = tf.placeholder(tf.float32, [None, n_input], name='X')
    y = tf.placeholder(tf.float32, [None], name='y')
    labels = tf.placeholder(tf.int32, [None], name='labels')
    lr = tf.placeholder(tf.float32, name='lr')
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    with tf.variable_scope('dnn'):
        # Declare tensorflow constants (hyperparameters)
        hidden_size, resolution, std_margin = hp['hidden_size'], tf.constant(hp['output_size'], dtype=tf.float32), tf.constant(hp['duration_std_margin'], dtype=tf.float32)
        # Define DNN layers
        wreg = _max_norm_regularizer(hp.get('max_norm_threshold'), 'max_norm')
        layer = _dense_layer(X, (n_input, hidden_size), dropout_keep_prob, 'input_layer', batch_norm=False, summarize=summarize_parameters, weights_regularizer=wreg)
        for i in range(1, hp['depth'] - 1):
            layer = _dense_layer(layer, (hidden_size, hidden_size), dropout_keep_prob, 'layer_' + str(i), summarize=summarize_parameters, weights_regularizer=wreg)
        logits = _dense_layer(layer, (hidden_size, hp['output_size']), 1., 'output_layer', summarize=summarize_parameters, activation=None, weights_regularizer=wreg)
    
    # Define loss and optimizer
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    pred = _buckets_to_duration(logits, bucket_means)
    rmse = tf.sqrt(tf.losses.mean_squared_error(y, pred))
    opt_algorithm = hp['opt']['algo']
    optimizer = opt_algorithm(learning_rate=lr) if opt_algorithm is not tf.train.MomentumOptimizer else opt_algorithm(learning_rate=lr, momentum=hp['opt']['m'])
    grads_and_vars = optimizer.compute_gradients(loss)
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
    return pred, rmse, optimize, (X, y, labels, lr, dropout_keep_prob), tf.train.Saver(), init_op

def train(model, dataset, train_labels, test_labels, hp, save_dir, predset=None):
    # Unpack parameters
    train_data, test_data, train_targets, test_targets = dataset
    pred, rmse, optimizer, placeholders, saver, init_op = model
    X, y, labels, lr, dropout_keep_prob = placeholders

    # Start tensorflow session
    with tf.Session(config=utils.tf_config(ALLOW_GPU_MEM_GROWTH)) as sess:
        sess.run(init_op)

        # Create summary utils
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
                batch_ys = train_targets[indices]
                batch_ls = train_labels[indices]
                sess.run(optimizer, feed_dict={X: batch_xs, y: batch_ys, labels: batch_ls, lr: hp['lr'], dropout_keep_prob: hp['dropout_keep_prob']})
                # Apply max norm regularization
                sess.run(max_norm_ops)
            # Evaluate model and display progress
            steps_since_improvement += 1
            if epoch % DISPLAY_STEP_PREDIOD == 0:
                summary, test_rmse = sess.run([summary_op, rmse], feed_dict={X: test_data, y: test_targets, labels: test_labels, dropout_keep_prob: 1.})
                summary_writer.add_summary(summary, epoch)
                if best_rmse > test_rmse:
                    steps_since_improvement = 0
                    best_rmse = test_rmse
                    print('Saving model...')
                    saver.save(sess, save_dir)
                print("Epoch=%03d/%03d, test_rmse=%.6f" % (epoch, hp['epochs'], test_rmse))
                if steps_since_improvement >= hp['early_stopping']:
                    print('Early stopping.')
                    break
        print("Training done, best_rmse=%.6f" % best_rmse)

        # Restore best model and apply prediction
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
    source_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = '/output/' if tf.flags.FLAGS.floyd_job else os.path.join(source_dir, 'models/')
    dataset_dir = '/input/' if tf.flags.FLAGS.floyd_job else os.path.join(source_dir, 'NYC_taxi_data_2016/')
   
    # Set log level to debug
    tf.logging.set_verbosity(tf.logging.INFO)

    # Parse and preprocess data
    knn_files = ('knn_train_features.npz', 'knn_test_features.npz', 'knn_pred_features.npz')
    features_len, (pred_ids, predset), dataset = load_data(dataset_dir, 'train.csv', 'test.csv', *knn_files)
    
    # Get buckets from train targets
    hyperparameters = DEFAULT_HYPERPARAMETERS
    train_labels, test_labels, bucket_means = get_buckets(dataset[2], dataset[3], hyperparameters['output_size'])

    # Build model
    print('Hyperparameters:\n' + str(hyperparameters))
    model = build_model(features_len, hyperparameters, bucket_means)

    # Train model
    print('Model built, starting training.')
    test_rmse, predictions = train(model, dataset, train_labels, test_labels, hyperparameters, save_dir, predset)

    # Save predictions to csv file for Kaggle submission
    predictions = np.int32(np.round(np.exp(predictions))) - 1
    pd.DataFrame(np.column_stack([pred_ids, predictions]), columns=['id', 'trip_duration']).to_csv(os.path.join(save_dir, 'preds.csv'), index=False)

if __name__ == '__main__':
    tf.flags.DEFINE_bool('floyd-job', False, 'Change working directories for training on Floyd.')
    tf.app.run()
