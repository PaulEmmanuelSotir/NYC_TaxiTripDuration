#!/usr/bin/python
# -*- coding: utf-8 -*-
""" NYC Taxi Trip Duration - Kaggle competion
Note that the code could probably be greatly simplified using tf.train.Supervisor and tf.contrib.learn.dnnregressor,
but we prefer to define model by hand here to learn more about tensorflow python API (and for more flexibility).

TODO:
    * use CV
    * improve early stoping (if windowed mean doesn't improve + compare with previous best rmse curves)

.. See https://github.com/PaulEmmanuelSotir/NYC_TaxiTripDuration
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import utils

__all__ = ['load_data', 'bucketize', 'build_model', 'train']

DEFAULT_HYPERPARAMETERS = {'epochs': 650,
                           'initial_cycle_length': 10,
                           'lr_cycle_growth': 2,
                           'initial_lr': 0.1,
                           'minimal_lr': 5e-8,
                           'keep_best_snapshot': 3,
                           'depth': 10,
                           'hidden_size': 512,
                           'batch_size': 1024,
                           'early_stopping': None,
                           'dropout_keep_prob': 1.,
                           'l2_regularization': 1.5e-4,
                           'output_size': 512}

TEST_SIZE = 100000
EVALUATE_PERIOD = 1
PRED_BATCH_SIZE = 32 * 1024
ALLOW_GPU_MEM_GROWTH = True
EXTENDED_SUMMARY_PERIOD = 40


def bucketize(train_targets, test_targets, bucket_count):
    """ Process buckets from train targets and deduce labels of trainset and testset """
    sorted_targets = np.sort(train_targets)
    bucket_size = len(sorted_targets) // bucket_count
    buckets = [sorted_targets[i * bucket_size: (1 + i) * bucket_size] for i in range(bucket_count)]
    # Bucketize targets (TODO: try soft classes)
    bucket_maxs = [np.max(b) for b in buckets]
    bucket_maxs[-1] = float('inf')

    def _find_indice(value): return np.searchsorted(bucket_maxs, value)
    train_labels = np.vectorize(_find_indice)(train_targets)
    test_labels = np.vectorize(_find_indice)(test_targets)
    # Process buckets means
    buckets_means = tf.constant([np.mean(bucket) for bucket in buckets], dtype=tf.float32, name='buckets_means')
    return train_labels, test_labels, buckets_means


def _buckets_to_duration(logits, bucket_means):
    return tf.reduce_sum(bucket_means * tf.nn.softmax(logits), axis=1)


def load_data(dataset_dir, train_file, test_file):
    """ Load data, add engineered features and split dataset into trainset, testset and predset """
    # TODO: put this code in feature_engineering.py
    from feature_engineering import load_features
    data, pred_data, targets, pred_ids = load_features(os.path.join(dataset_dir, train_file), os.path.join(dataset_dir, test_file))

    # Split dataset into trainset and testset
    train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=TEST_SIZE, random_state=459)

    # Normalize feature columns
    standardizer = preprocessing.StandardScaler()
    train_data = standardizer.fit_transform(train_data)
    test_data = standardizer.transform(test_data)
    pred_data = standardizer.transform(pred_data)

    return pred_data.shape[1], (pred_ids, pred_data), (train_data, test_data, train_targets, test_targets)


def _dense_layer(x, shape, dropout_keep_prob, name, batch_norm=True, summarize=True, activation=tf.nn.tanh, weights_regularizer=None, training=False):
    with tf.variable_scope(name):
        weights = tf.Variable(utils.xavier_init(*shape, activation='tanh'), name='w')
        if weights_regularizer is not None:
            weights_regularizer(weights)
        bias = tf.Variable(tf.truncated_normal([shape[1]]) if shape[1] > 1 else 0., name='b')
        logits = tf.add(tf.matmul(x, weights), bias)
        logits_bn = tf.layers.batch_normalization(logits, training=training) if batch_norm else logits
        dense = activation(logits_bn) if activation is not None else logits_bn
        dense_do = dense if dropout_keep_prob == 1. else tf.nn.dropout(dense, dropout_keep_prob)
        if summarize:
            image = tf.reshape(weights, [1, weights.shape[0].value, weights.shape[1].value, 1])
            tf.summary.image('weights', image, collections=['extended_summary'])
            tf.summary.histogram('weights_histogram', weights, collections=['extended_summary'])
            tf.summary.histogram('bias', bias, collections=['extended_summary'])
    return dense_do, weights


def build_model(n_input, hp, bucket_means, summarize_parameters=True):
    """ Define Tensorflow DNN model architechture """
    # Input placeholders
    with tf.name_scope('inputs'):
        lr = tf.placeholder(tf.float32, [], name='learning_rate')
        labels = tf.placeholder(tf.int32, [None], name='labels')
        dropout_keep_prob = tf.placeholder(tf.float32, [], name='dropout_keep_prob')
        X = tf.placeholder(tf.float32, [None, n_input], name='X')
        y = tf.placeholder(tf.float32, [None], name='y')
        l2_regularization = tf.placeholder(tf.float32, [], name='l2_regularization')
        training = tf.placeholder_with_default(False, [], name='training')

    weights = []
    with tf.variable_scope('dnn'):
        # Define DNN layers
        layer, w = _dense_layer(X, (n_input, hp['hidden_size']), dropout_keep_prob, 'input_layer',
                                batch_norm=False, summarize=summarize_parameters, training=training)
        weights.append(w)
        for i in range(1, hp['depth'] - 1):
            layer, w = _dense_layer(layer, (hp['hidden_size'], hp['hidden_size']), dropout_keep_prob,
                                    'layer_' + str(i), summarize=summarize_parameters, training=training)
            weights.append(w)
        logits, w = _dense_layer(layer, (hp['hidden_size'], hp['output_size']), 1., 'output_layer',
                                 summarize=summarize_parameters, activation=None, training=training)
        weights.append(w)

    # Define loss and optimizer
    pred = _buckets_to_duration(logits, bucket_means)
    rmse = tf.sqrt(tf.losses.mean_squared_error(y, pred), name='rmse')
    with tf.name_scope('L2_regularization'):
        L2 = l2_regularization * tf.add_n([tf.nn.l2_loss(w) for w in weights])
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits) + L2
    tf.summary.histogram('pred', pred, collections=['extended_summary'])
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True)
    grads_and_vars = optimizer.compute_gradients(loss)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimize = optimizer.apply_gradients(grads_and_vars)

    # Variable initialization operation
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    return pred, rmse, loss, optimize, (X, y, labels, lr, dropout_keep_prob, l2_regularization, training), tf.train.Saver(), init_op


def train(model, dataset, train_labels, test_labels, hp, save_dir, predset):
    # Unpack parameters
    train_data, test_data, train_targets, test_targets = dataset
    pred, rmse, loss, optimizer, placeholders, saver, init_op = model
    X, y, labels, lr, dropout_keep_prob, l2_regularization, training = placeholders

    # Start tensorflow session
    with tf.Session(config=utils.tf_config(ALLOW_GPU_MEM_GROWTH)) as sess:
        sess.run(init_op)

        # Create summary utils
        extended_summary_op = tf.summary.merge_all('extended_summary')
        summary_writer = tf.summary.FileWriter(save_dir, sess.graph)

        # Training loop
        cycle = 0
        steps_since_improvement = 0
        batch_per_epoch = len(train_data) // hp['batch_size']
        best_rmse = float('inf')
        for epoch in range(hp['epochs']):
            # Train model using minibatches
            mean_rmse, mean_loss, mean_lr = 0., 0., 0.
            learning_rate = hp['initial_lr']
            for batch in range(batch_per_epoch):
                indices = np.random.randint(len(train_data), size=hp['batch_size'])  # TODO: create batches using tf.batch instead
                batch_rmse, batch_loss, _ = sess.run([rmse, loss, optimizer], feed_dict={X: train_data[indices, :],
                                                                                         y: train_targets[indices],
                                                                                         labels: train_labels[indices],
                                                                                         lr: learning_rate,
                                                                                         dropout_keep_prob: hp['dropout_keep_prob'],
                                                                                         l2_regularization: hp['l2_regularization'],
                                                                                         training: True})
                learning_rate, new_cycle = utils.warm_restart(epoch + batch / batch_per_epoch, hp['initial_cycle_length'],
                                                              max_lr=hp['initial_lr'], min_lr=hp['minimal_lr'], t_mult=hp['lr_cycle_growth'])
                if new_cycle:
                    print('Saving cycle #' + str(cycle) + ' snapshot...')
                    saver.save(sess, os.path.join(save_dir, str(cycle % hp['keep_best_snapshot'])) + '/')
                    cycle += 1

                def _moving_mean(value): return len(indices) * value / len(train_data)
                mean_rmse += _moving_mean(batch_rmse)
                mean_loss += _moving_mean(batch_loss)
                mean_lr += _moving_mean(learning_rate)
            utils.add_summary_values(summary_writer, global_step=epoch + 1, mean_rmse=mean_rmse, mean_loss=mean_loss, mean_lr=mean_lr)
            # Evaluate model and display progress
            if epoch % EVALUATE_PERIOD == 0:
                test_rmse = sess.run(rmse, feed_dict={X: test_data, y: test_targets, labels: test_labels, dropout_keep_prob: 1.})
                utils.add_summary_values(summary_writer, global_step=epoch + 1, test_rmse=test_rmse)
                steps_since_improvement += 1
                if best_rmse > test_rmse:
                    print('Best test_rmse encountered so far, saving model...')
                    best_rmse = test_rmse
                    steps_since_improvement = 0
                    saver.save(sess, save_dir)
                print("Epoch=%03d/%03d, test_rmse=%.6f" % (epoch + 1, hp['epochs'], test_rmse))
                if hp.get('early_stopping') is not None and steps_since_improvement >= hp['early_stopping']:
                    print('Early stopping.')
                    break
            if epoch % EXTENDED_SUMMARY_PERIOD == 0:
                summary = sess.run(extended_summary_op, feed_dict={X: test_data, dropout_keep_prob: 1.})
                summary_writer.add_summary(summary, epoch + 1)
        print("Training done, best_rmse=%.6f" % best_rmse)

        # Restore best model and apply prediction
        test_rmse = []
        predictions = []
        for snapshot in range(hp['keep_best_snapshot']):
            saver.restore(sess, os.path.join(save_dir, str(snapshot)) + '/')
            print('Applying snapshot #' + str(snapshot))
            test_rmse.append(sess.run(pred, feed_dict={X: test_data, dropout_keep_prob: 1.}))
            predictions.append([])
            for batch in np.array_split(predset, len(predset) // PRED_BATCH_SIZE):
                predictions[-1].extend(sess.run(pred, feed_dict={X: batch, dropout_keep_prob: 1.}))
        predictions = np.mean(predictions, axis=0)
        print('Ensemble test_rmse=' + str(np.sqrt(np.mean((np.mean(test_rmse, axis=0) - test_targets) ** 2))))
    return best_rmse, predictions


def main(_=None):
    # Define working directories
    source_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = '/output/' if tf.flags.FLAGS.floyd_job else os.path.join(source_dir, 'models/')
    dataset_dir = '/input/' if tf.flags.FLAGS.floyd_job else os.path.join(source_dir, 'NYC_taxi_data_2016/')
    hyperparameters = DEFAULT_HYPERPARAMETERS
    print('Hyperparameters:\n' + str(hyperparameters))

    # Parse and preprocess data
    features_len, (pred_ids, predset), dataset = load_data(dataset_dir, 'train.csv', 'test.csv')

    # Get buckets from train targets
    (_, _, train_targets, test_targets) = dataset
    train_labels, test_labels, bucket_means = bucketize(train_targets, test_targets, hyperparameters['output_size'])

    # Build model
    model = build_model(features_len, hyperparameters, bucket_means)

    # Train model
    print('Model built, starting training.')
    _, predictions = train(model, dataset, train_labels, test_labels, hyperparameters, save_dir, predset)

    # Save predictions to csv file for Kaggle submission
    predictions = np.int32(np.round(np.exp(predictions))) - 1
    pd.DataFrame(np.column_stack([pred_ids, predictions]), columns=['id', 'trip_duration']).to_csv(os.path.join(save_dir, 'preds.csv'), index=False)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)  # Set log level to debug
    tf.flags.DEFINE_bool('floyd-job', False, 'Change working directories for training on Floyd.')
    tf.app.run()
