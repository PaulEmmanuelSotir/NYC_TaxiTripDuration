#!/usr/bin/python
# -*- coding: utf-8 -*-
""" NYC Taxi Trip Duration - Kaggle competition
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

import feature_engineering
import utils

__all__ = ['bucketize', 'build_model', 'train']

DEFAULT_HYPERPARAMETERS = {'epochs': 610,
                           'lr': 0.1,
                           'warm_resart_lr': {
                               'initial_cycle_length': 20,
                               'lr_cycle_growth': 1.5,
                               'minimal_lr': 5e-8,
                               'keep_best_snapshot': 3
                           },
                           'depth': 10,
                           'hidden_size': 512,
                           'batch_size': 1024,
                           'early_stopping': None,
                           'dropout_keep_prob': 1.,
                           'l2_regularization': 5e-4,
                           'output_size': 512}

VALID_SIZE = 100000
EVALUATE_PERIOD = 1
PRED_BATCH_SIZE = 32 * 1024
ALLOW_GPU_MEM_GROWTH = True
EXTENDED_SUMMARY_PERIOD = 40


def bucketize(train_targets, valid_targets, bucket_count):
    """ Process buckets from train targets and deduce labels of trainset and testset """
    sorted_targets = np.sort(train_targets)
    bucket_size = len(sorted_targets) // bucket_count
    buckets = [sorted_targets[i * bucket_size: (1 + i) * bucket_size] for i in range(bucket_count)]
    # Bucketize targets (TODO: try soft classes)
    bucket_maxs = [np.max(b) for b in buckets]
    bucket_maxs[-1] = float('inf')

    def _find_indice(value): return np.searchsorted(bucket_maxs, value)
    train_labels = np.vectorize(_find_indice)(train_targets)
    valid_labels = np.vectorize(_find_indice)(valid_targets)
    # Process buckets means
    buckets_means = tf.constant([np.mean(bucket) for bucket in buckets], dtype=tf.float32, name='buckets_means')
    return train_labels, valid_labels, buckets_means


def _dense_layer(x, shape, dropout_keep_prob, name, batch_norm=True, summarize=True, activation=tf.nn.tanh, training=False):
    with tf.variable_scope(name):
        weights = tf.Variable(utils.xavier_init(*shape, activation='tanh'), name='w')
        bias = tf.Variable(tf.truncated_normal([shape[1]]) if shape[1] > 1 else 0., name='b')
        logits = tf.add(tf.matmul(x, weights), bias)
        logits_bn = tf.layers.batch_normalization(logits, training=training) if batch_norm else logits
        dense = activation(logits_bn) if activation is not None else logits_bn
        dense_do = tf.nn.dropout(dense, dropout_keep_prob)
        if summarize:
            image = tf.reshape(weights, [1, weights.shape[0].value, weights.shape[1].value, 1])
            tf.summary.image('weights', image, collections=['extended_summary'])
            tf.summary.histogram('weights_histogram', weights, collections=['extended_summary'])
            tf.summary.histogram('bias', bias, collections=['extended_summary'])
    return dense_do, weights


def build_model(n_input, hp, bucket_means, summarize=True):
    """ Define Tensorflow DNN model architechture """
    # Input placeholders
    with tf.name_scope('inputs'):
        lr = tf.placeholder(tf.float32, [], name='learning_rate')
        labels = tf.placeholder(tf.int32, [None], name='labels')
        dropout_keep_prob = tf.placeholder_with_default(1., [], name='dropout_keep_prob')
        X = tf.placeholder(tf.float32, [None, n_input], name='X')
        y = tf.placeholder(tf.float32, [None], name='y')
        l2_regularization = tf.placeholder(tf.float32, [], name='l2_regularization')
        train = tf.placeholder_with_default(False, [], name='training')

    weights = []
    with tf.variable_scope('dnn'):
        # Define DNN layers
        layer, w = _dense_layer(X, (n_input, hp['hidden_size']), dropout_keep_prob, 'input_layer', batch_norm=False, summarize=summarize, training=train)
        weights.append(w)
        for i in range(1, hp['depth'] - 1):
            layer, w = _dense_layer(layer, (hp['hidden_size'], hp['hidden_size']), dropout_keep_prob, 'layer_' + str(i), summarize=summarize, training=train)
            weights.append(w)
        logits, w = _dense_layer(layer, (hp['hidden_size'], hp['output_size']), 1., 'output_layer', summarize=summarize, activation=None, training=train)
        weights.append(w)

    # Define loss and optimizer

    pred = tf.reduce_sum(bucket_means * tf.nn.softmax(logits), axis=1)
    rmse = tf.sqrt(tf.losses.mean_squared_error(y, pred), name='rmse')
    with tf.name_scope('L2_regularization'):
        L2 = l2_regularization * tf.add_n([tf.nn.l2_loss(w) for w in weights])
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits) + L2
    tf.summary.histogram('pred', pred, collections=['extended_summary'])
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, use_nesterov=True, momentum=0.9)
    grads_and_vars = optimizer.compute_gradients(loss)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimize = optimizer.apply_gradients(grads_and_vars)

    # Variable initialization operation
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    return pred, rmse, loss, optimize, (X, y, labels, lr, dropout_keep_prob, l2_regularization, train), tf.train.Saver(), init_op


def train(model, dataset, train_labels, valid_labels, hp, save_dir, testset):
    # Unpack parameters
    train_data, valid_data, train_targets, valid_targets = dataset
    pred, rmse, loss, optimizer, placeholders, saver, init_op = model
    X, y, labels, lr, dropout_keep_prob, l2_regularization, training = placeholders
    wr_hp = hp.get('warm_resart_lr')

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
            learning_rate = hp['lr']
            for batch in range(batch_per_epoch):
                indices = np.random.randint(len(train_data), size=hp['batch_size'])  # TODO: create batches using tf.batch instead
                batch_rmse, batch_loss, _ = sess.run([rmse, loss, optimizer], feed_dict={X: train_data[indices, :],
                                                                                         y: train_targets[indices],
                                                                                         labels: train_labels[indices],
                                                                                         lr: learning_rate,
                                                                                         dropout_keep_prob: hp['dropout_keep_prob'],
                                                                                         l2_regularization: hp['l2_regularization'],
                                                                                         training: True})
                if wr_hp is not None:
                    learning_rate, new_cycle = utils.warm_restart(epoch + (1 + batch) / batch_per_epoch, t_0=wr_hp['initial_cycle_length'],
                                                                  max_lr=hp['lr'], min_lr=wr_hp['minimal_lr'], t_mult=wr_hp['lr_cycle_growth'])
                    if new_cycle:
                        print('Saving cycle #' + str(cycle) + ' snapshot...')
                        saver.save(sess, os.path.join(save_dir, str(cycle % wr_hp['keep_best_snapshot'])) + '/')
                        cycle += 1

                def _moving_mean(value): return len(indices) * value / len(train_data)
                mean_rmse += _moving_mean(batch_rmse)
                mean_loss += _moving_mean(batch_loss)
                mean_lr += _moving_mean(learning_rate)
            utils.add_summary_values(summary_writer, global_step=epoch + 1, mean_rmse=mean_rmse, mean_loss=mean_loss, mean_lr=mean_lr)
            # Evaluate model and display progress
            if epoch % EVALUATE_PERIOD == 0:
                valid_rmse = sess.run(rmse, feed_dict={X: valid_data, y: valid_targets, labels: valid_labels})
                utils.add_summary_values(summary_writer, global_step=epoch + 1, valid_rmse=valid_rmse)
                steps_since_improvement += 1
                if best_rmse > valid_rmse:
                    print('Best valid_rmse encountered so far, saving model...')
                    best_rmse = valid_rmse
                    steps_since_improvement = 0
                    saver.save(sess, save_dir)
                print("Epoch=%03d/%03d, valid_rmse=%.6f" % (epoch + 1, hp['epochs'], valid_rmse))
                if hp.get('early_stopping') is not None and steps_since_improvement >= hp['early_stopping']:
                    print('Early stopping.')
                    break
            if epoch % EXTENDED_SUMMARY_PERIOD == 0:
                summary = sess.run(extended_summary_op, feed_dict={X: valid_data})
                summary_writer.add_summary(summary, epoch + 1)
        print("Training done, best_rmse=%.6f" % best_rmse)

        ensemble_rmse = float('inf')
        if wr_hp is not None:
            # Apply last cycle snapshots on test and validation set and average softmax layers for prediction (snaphot ensembling)
            valid_preds = []
            predictions = []
            for snapshot in range(wr_hp['keep_best_snapshot']):
                saver.restore(sess, os.path.join(save_dir, str(snapshot)) + '/')
                print('Applying snapshot #' + str(snapshot))
                valid_preds.append(sess.run(pred, feed_dict={X: valid_data}))
                predictions.append([])
                for batch in np.array_split(testset, len(testset) // PRED_BATCH_SIZE):
                    predictions[-1].extend(sess.run(pred, feed_dict={X: batch}))
            predictions = np.mean(predictions, axis=0)
            ensemble_rmse = np.sqrt(np.mean((np.mean(valid_preds, axis=0) - valid_targets) ** 2))
            print('Ensemble valid_rmse=' + str(ensemble_rmse))
        if ensemble_rmse > best_rmse:
            # Restore best model snapshot and apply prediction
            print("Snaphsot ensemble isn't enabled or didn't performed better than best rsmse encountered during training")
            print('Applying best_rmse snapshot on test set')
            predictions = []
            saver.restore(sess, save_dir)
            for batch in np.array_split(testset, len(testset) // PRED_BATCH_SIZE):
                predictions.extend(sess.run(pred, feed_dict={X: batch}))
    return best_rmse, predictions


def main(_=None):
    # Define working directories
    source_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = '/output/' if tf.flags.FLAGS.floyd_job else os.path.join(source_dir, 'models/')
    dataset_dir = '/input/' if tf.flags.FLAGS.floyd_job else os.path.join(source_dir, 'NYC_taxi_data_2016/')
    hyperparameters = DEFAULT_HYPERPARAMETERS
    print('Hyperparameters:\n' + str(hyperparameters))

    # Parse and preprocess data
    features_len, (test_ids, testset), dataset = feature_engineering.load_data(dataset_dir, 'train.csv', 'test.csv', VALID_SIZE, cache_read_only=tf.flags.FLAGS.floyd_job)

    # Get buckets from train targets
    (_, _, train_targets, valid_targets) = dataset
    train_labels, valid_labels, bucket_means = bucketize(train_targets, valid_targets, hyperparameters['output_size'])

    # Build model
    model = build_model(features_len, hyperparameters, bucket_means)

    # Train model
    print('Model built, starting training.')
    _, test_preds = train(model, dataset, train_labels, valid_labels, hyperparameters, save_dir, testset)

    # Save predictions to csv file for Kaggle submission
    test_preds = np.int32(np.round(np.exp(test_preds))) - 1
    pd.DataFrame(np.column_stack([test_ids, test_preds]), columns=['id', 'trip_duration']).to_csv(os.path.join(save_dir, 'preds.csv'), index=False)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)  # Set log level to debug
    tf.flags.DEFINE_bool('floyd-job', False, 'Change working directories for training on Floyd.')
    tf.app.run()
