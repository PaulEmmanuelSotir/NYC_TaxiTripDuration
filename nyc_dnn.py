#!/usr/bin/python
# -*- coding: utf-8 -*-
""" NYC Taxi Trip Duration - Kaggle competition
Note that the code could probably be greatly simplified using tf.train.Supervisor and tf.contrib.learn.dnnregressor,
but we prefer to define model by hand here to learn more about tensorflow python API (and for more flexibility).

.. See https://github.com/PaulEmmanuelSotir/NYC_TaxiTripDuration
"""
import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf

import feature_engineering
import utils

__all__ = ['build_graph', 'train']

DEFAULT_HYPERPARAMETERS = {'epochs': 96,
                           'lr': 0.1,
                           'warm_resart_lr': {
                               'initial_cycle_length': 20,
                               'lr_cycle_growth': 1.5,
                               'minimal_lr': 5e-8,
                               'keep_best_snapshot': 2},
                           'depth': 10,
                           'hidden_size': 512,
                           'batch_size': 1024,
                           'early_stopping': None,
                           'dropout_keep_prob': 1.,
                           'l2_regularization': 2.5e-4,
                           'output_size': 512}

VALID_SIZE = 100000
EVALUATE_PERIOD = 1
PRED_BATCH_SIZE = 64 * 1024
ALLOW_GPU_MEM_GROWTH = True
EXTENDED_SUMMARY_EVAL_PERIOD = 40


def _dense_layer(x, shape, dropout_keep_prob, name, batch_norm=True, summarize=True, activation=tf.nn.tanh, training=False):
    with tf.variable_scope(name):
        weights = tf.get_variable(initializer=utils.xavier_init('tanh')(shape), name='w',
                                  collections=['weights', tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES])
        bias = tf.get_variable(initializer=tf.truncated_normal([shape[1]]) if shape[1] > 1 else 0., name='b')
        logits = tf.add(tf.matmul(x, weights), bias)
        logits_bn = tf.layers.batch_normalization(logits, training=training) if batch_norm else logits
        dense = activation(logits_bn) if activation is not None else logits_bn
        dense_do = tf.nn.dropout(dense, dropout_keep_prob)
        if summarize:
            image = tf.reshape(weights, [1, weights.shape[0].value, weights.shape[1].value, 1])
            tf.summary.image('weights', image, collections=['extended_summary'])
            tf.summary.histogram('weights_histogram', weights, collections=['extended_summary'])
            tf.summary.histogram('bias', bias, collections=['extended_summary'])
    return dense_do


def _conv_layer(x, filters, kernel_size, dropout_keep_prob, name, batch_norm=True, summarize=True, activation=tf.nn.elu, training=False):
    with tf.variable_scope(name):
        conv = tf.layers.conv1d(x, filters=filters, kernel_size=kernel_size, strides=1, padding='same', activation=None,
                                kernel_initializer=utils.xavier_init('relu'))
        conv = tf.layers.batch_normalization(conv, training=training) if batch_norm else conv
        conv = activation(conv) if activation is not None else conv
        conv = tf.nn.dropout(conv, dropout_keep_prob)
        """
        kernel = tf.get_variable('conv1d/kernel:0')
        if summarize:
            bias = ...
            image = tf.reshape(kernels, [...])
            tf.summary.image('kernels', image, collections=['extended_summary'])
            tf.summary.histogram('bias', bias, collections=['extended_summary'])"""
    return conv


def _build_dnn(X, n_input, hp, bucket_means, dropout_keep_prob, summarize, training=False):
    """ Define Tensorflow DNN model architechture """
    hidden_size = hp['hidden_size']
    with tf.variable_scope('dnn'):
        # Define DNN layers
        layer = _dense_layer(X, (n_input, hidden_size), dropout_keep_prob, 'input_layer', batch_norm=False, summarize=summarize, training=training)
        for i in range(1, hp['depth'] - 1):
            layer = _dense_layer(layer, (hidden_size, hidden_size), dropout_keep_prob, 'layer_' + str(i), summarize=summarize, training=training)
        logits = _dense_layer(layer, (hidden_size, hp['output_size']), 1., 'output_layer', summarize=summarize, activation=None, training=training)
    pred = tf.reduce_sum(bucket_means * tf.nn.softmax(logits), axis=1)
    return pred, logits


def _build_cnn(X, n_input, hp, bucket_means, dropout_keep_prob, summarize, training=False):
    hidden_size = hp['hidden_size']
    with tf.variable_scope('cnn'):
        # Define input fully connected layers
        net = _dense_layer(X, (n_input, hidden_size), dropout_keep_prob, 'input_layer_1', batch_norm=False, summarize=summarize, training=training)
        net = _dense_layer(net, (hidden_size, hidden_size), dropout_keep_prob, 'input_layer_2', summarize=summarize, training=training)

        # Define convolutionnal layers
        filters = 8
        net = tf.reshape(net, shape=[-1, hidden_size, 1])
        for i in range(1, hp['depth'] - 3):
            net = _conv_layer(net, filters, 4, dropout_keep_prob, name='conv_' + str(i), summarize=summarize, training=training)
        net = tf.reshape(net, shape=[-1, filters * hidden_size])

        # Define output fully connected layers
        net = _dense_layer(net, (filters * hidden_size, hidden_size), dropout_keep_prob, 'output_layer_1', summarize=summarize, training=training)
        logits = _dense_layer(net, (hidden_size, hidden_size), 1., 'output_layer_2', activation=None, summarize=summarize, training=training)
    pred = tf.reduce_sum(bucket_means * tf.nn.softmax(logits), axis=1)
    return pred, logits


def build_graph(n_input, hp, bucket_means, summarize=True):
    """ Build Tensorflow training, validation and testing graph """
    # Define inputs
    with tf.variable_scope('inputs'):
        lr = tf.placeholder(tf.float32, [], name='learning_rate')
        dropout_keep_prob = tf.placeholder_with_default(1., [], name='dropout_keep_prob')
        l2_reg = tf.placeholder(tf.float32, [], name='l2_regularization')
        X = tf.placeholder(tf.float32, [None, n_input], name='X')
        labels = tf.placeholder(tf.int32, [None], name='labels')
        y = tf.placeholder(tf.float32, [None], name='y')
        train = tf.placeholder_with_default(False, [], name='training')

    bucket_means = tf.constant(bucket_means, dtype=tf.float32, name='buckets_means')
    pred, logits = _build_dnn(X, n_input, hp, bucket_means, dropout_keep_prob, summarize=summarize, training=train)
    tf.summary.histogram('pred', pred, collections=['extended_summary'])
    rmse = tf.sqrt(tf.losses.mean_squared_error(y, pred), name='rmse')

    # Define loss and optimizer
    with tf.variable_scope('L2_regularization'):
        L2 = l2_reg * tf.add_n([tf.nn.l2_loss(w) for w in tf.get_collection('weights')])
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) + L2
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, use_nesterov=True, momentum=0.9)
    grads_and_vars = optimizer.compute_gradients(loss)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        apply_grad = optimizer.apply_gradients(grads_and_vars)
    train_op = tf.tuple([rmse, loss], control_inputs=[apply_grad], name='train_ops')

    # Variable initialization operation
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), name='init_op')
    return pred, rmse, loss, train_op, (lr, dropout_keep_prob, l2_reg, X, labels, y, train), tf.train.Saver(), init_op


def train(model, dataset, hp, save_dir, testset):
    # Unpack parameters and tensors
    train_data, valid_data, train_targets, valid_targets, train_labels, _ = dataset
    pred, rmse, loss, train_op, placeholders, saver, init_op = model
    lr, dropout_keep_prob, l2_regularization, X, labels, y, training = placeholders
    wr_hp = hp.get('warm_resart_lr')

    # Start tensorflow session
    with tf.Session(config=utils.tf_config(ALLOW_GPU_MEM_GROWTH)) as sess:
        sess.run(init_op)

        # Create summary utils
        extended_summary_op = tf.summary.merge_all('extended_summary')
        summary_writer = tf.summary.FileWriter(save_dir, sess.graph)

        # Training loop
        best_rmse = float('inf')
        learning_rate = hp['lr']
        mean_rmse, mean_loss, mean_lr = 0., 0., 0.
        cycle, epochs_since_improvement = 0, 0
        batch_per_epoch = int(np.ceil(len(train_data) / hp['batch_size']))
        for epoch in range(hp['epochs']):
            # Shuffle trainset
            perm = np.random.permutation(len(train_data))
            train_data, train_labels, train_targets = (train_data[perm], train_labels[perm], train_targets[perm])
            # Train model using minibatches
            mean_rmse, mean_loss, mean_lr = 0., 0., 0.
            for step, range_min in zip(range(batch_per_epoch), range(0, len(train_data) - 1, hp['batch_size'])):
                range_max = min(range_min + hp['batch_size'], len(train_data))
                batch_rmse, batch_loss = sess.run(train_op, feed_dict={X: train_data[range_min:range_max],
                                                                       y: train_targets[range_min:range_max],
                                                                       labels: train_labels[range_min:range_max],
                                                                       lr: learning_rate,
                                                                       dropout_keep_prob: hp['dropout_keep_prob'],
                                                                       l2_regularization: hp['l2_regularization'],
                                                                       training: True})
                if wr_hp is not None:
                    learning_rate, new_cycle = utils.warm_restart(epoch + step / batch_per_epoch, t_0=wr_hp['initial_cycle_length'],
                                                                  max_lr=hp['lr'], min_lr=wr_hp['minimal_lr'], t_mult=wr_hp['lr_cycle_growth'])
                    if new_cycle:
                        print('New learning rate cycle.')
                        epochs_since_improvement = 0  # We reset early stoping for each lr cycles
                        cycle += 1
                mean_rmse += (range_max - range_min) * batch_rmse / len(train_data)
                mean_loss += (range_max - range_min) * batch_loss / len(train_data)
                mean_lr += (range_max - range_min) * learning_rate / len(train_data)

            # Evaluate model and display progress
            utils.add_summary_values(summary_writer, global_step=epoch, mean_rmse=mean_rmse, mean_loss=mean_loss, mean_lr=mean_lr)
            if epoch % EVALUATE_PERIOD == 0:
                # Process RMSE on validation set
                if (epoch // EVALUATE_PERIOD) % EXTENDED_SUMMARY_EVAL_PERIOD == 0:
                    valid_rmse, summary = sess.run([rmse, extended_summary_op], feed_dict={X: valid_data, y: valid_targets})
                    summary_writer.add_summary(summary, epoch)
                else:
                    valid_rmse = sess.run(rmse, feed_dict={X: valid_data, y: valid_targets})
                utils.add_summary_values(summary_writer, global_step=epoch, valid_rmse=valid_rmse)
                # Save snapshot if validation RMSE is the best encountered so far
                if best_rmse > valid_rmse:
                    print('Best valid_rmse encountered so far, saving model...')
                    best_rmse = valid_rmse
                    best_saveddir = os.path.join(save_dir, 'best_snapshot/')
                    epochs_since_improvement = 0
                    saver.save(sess, best_saveddir)
                    if wr_hp:
                        # Copy snapshot to cycle's directory
                        snapshot_saveddir = os.path.join(save_dir, str(cycle % wr_hp['keep_best_snapshot']))
                        shutil.rmtree(snapshot_saveddir, ignore_errors=True)
                        shutil.copytree(best_saveddir, snapshot_saveddir)
                print("Epoch=%03d/%03d, valid_rmse=%.6f" % (epoch + 1, hp['epochs'], valid_rmse))
            if hp.get('early_stopping') is not None and epochs_since_improvement >= hp['early_stopping']:
                print('Early stopping.')
                break
            epochs_since_improvement += 1

        def _predict_testset():
            predictions = []
            for range_min in range(0, len(testset) - 1, PRED_BATCH_SIZE):
                batch_X = testset[range_min: min(range_min + PRED_BATCH_SIZE, len(testset))]
                predictions.extend(sess.run(pred, feed_dict={X: batch_X}))
            return predictions

        print("Training done, best_rmse=%.6f" % best_rmse)
        ensemble_rmse = float('inf')
        if wr_hp is not None:
            # Apply last cycle snapshots on test and validation set and average softmax layers for prediction (snaphot ensembling)
            predictions, valid_preds = [], []
            for snapshot in range(wr_hp['keep_best_snapshot']):
                print('Applying snapshot #' + str(snapshot))
                saver.restore(sess, os.path.join(save_dir, str(snapshot)) + '/')
                valid_preds.append(sess.run(pred, feed_dict={X: valid_data, y: valid_targets}))
                predictions.append(_predict_testset())

            predictions = np.mean(predictions, axis=0)
            ensemble_rmse = np.sqrt(np.mean((np.mean(valid_preds, axis=0) - valid_targets) ** 2))
            print('Ensemble valid_rmse=' + str(ensemble_rmse))
        if ensemble_rmse > best_rmse:
            # Restore best model snapshot and apply prediction
            print("Snaphsot ensemble isn't enabled or didn't performed better than best rsmse encountered during training")
            print('Applying best_rmse snapshot on test set')
            saver.restore(sess, os.path.join(save_dir, 'best_snapshot/'))
            predictions = _predict_testset()
    return best_rmse, predictions


def main(_=None):
    # Define working directories
    source_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = '/output/' if tf.flags.FLAGS.floyd_job else os.path.join(source_dir, 'models/')
    dataset_dir = '/input/' if tf.flags.FLAGS.floyd_job else os.path.join(source_dir, 'NYC_taxi_data_2016/')
    hyperparameters = DEFAULT_HYPERPARAMETERS
    print('Hyperparameters:\n' + str(hyperparameters))

    # Parse and preprocess data
    features_len, (test_ids, testset), dataset, bucket_means = feature_engineering.load_data(dataset_dir,
                                                                                             'train.csv',
                                                                                             'test.csv',
                                                                                             VALID_SIZE,
                                                                                             hyperparameters['output_size'],
                                                                                             tf.flags.FLAGS.floyd_job)

    # Build model
    model = build_graph(features_len, hyperparameters, bucket_means)

    # Train model
    print('Model built, starting training.')
    _, test_preds = train(model, dataset, hyperparameters, save_dir, testset)

    # Save predictions to csv file for Kaggle submission
    test_preds = np.int32(np.round(np.exp(test_preds))) - 1
    pd.DataFrame(np.column_stack([test_ids, test_preds]), columns=['id', 'trip_duration']).to_csv(os.path.join(save_dir, 'preds.csv'), index=False)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)  # Set log level to debug
    tf.flags.DEFINE_bool('floyd-job', False, 'Change working directories for training on Floyd.')
    tf.app.run()
