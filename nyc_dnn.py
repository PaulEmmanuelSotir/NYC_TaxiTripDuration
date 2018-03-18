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
from tensorflow.python.keras.layers import Embedding
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope

import feature_engineering
import utils

__all__ = ['build_graph', 'train']

DEFAULT_HYPERPARAMETERS = {'n_models': 1,
                           'epochs': 96,
                           'lr': 0.1,
                           'warm_resart_lr': {
                               'initial_cycle_length': 20,
                               'lr_cycle_growth': 1.5,
                               'minimal_lr': 5e-8,
                               'keep_best_snapshot': 2},
                           'momentum': 0.9,
                           'depth': 10,
                           'embedding_dim': 8,
                           'max_embedding_values': 64,
                           'hidden_size': 320,
                           'batch_size': 1024,
                           'early_stopping': None,
                           'dropout_keep_prob': 1.,
                           'l2_regularization': 2.4e-4,
                           'output_size': 512}

VALID_SIZE = 100000
EVALUATE_PERIOD = 1
PRED_BATCH_SIZE = 64 * 1024
ALLOW_GPU_MEM_GROWTH = True
EXTENDED_SUMMARY_EVAL_PERIOD = 40


@add_arg_scope
def _dense_layer(x, shape, dropout_keep_prob, name, batch_norm=True, summarize=True, activation=tf.nn.relu, init=utils.relu_xavier_avg, training=False, weights_col=None):
    with tf.variable_scope(name):
        collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES]
        if weights_col is not None:
            collections.append(weights_col)
        weights = tf.get_variable(initializer=init(shape), name='w', collections=collections)
        bias = tf.get_variable(initializer=tf.truncated_normal([shape[1]]) if shape[1] > 1 else 0., name='b')
        logits = tf.add(tf.matmul(x, weights), bias)
        if batch_norm:
            logits = tf.layers.batch_normalization(logits, training=training, name='batch_norm')
        dense = activation(logits) if activation is not None else logits
        dense = tf.nn.dropout(dense, dropout_keep_prob)
        if summarize:
            image = tf.reshape(weights, [1, weights.shape[0].value, weights.shape[1].value, 1])
            tf.summary.image('weights', image, collections=['extended_summary'])
            tf.summary.histogram('weights_histogram', weights, collections=['extended_summary'])
            tf.summary.histogram('bias', bias, collections=['extended_summary'])
    return dense


def _build_dnn(X, hp, bucket_means, dropout_keep_prob, summarize, training=False, weights_col='weights'):
    """ Define Tensorflow DNN model architechture """
    hidden_size = hp['hidden_size']
    with tf.variable_scope('dnn'):
        # Define DNN layers
        input_layer_dropout = 1. if 'embedding_dim' in hp else dropout_keep_prob
        with arg_scope([_dense_layer], summarize=summarize, training=training, weights_col=weights_col):
            layer = _dense_layer(X, (X.shape[-1].value, hidden_size), input_layer_dropout, 'input_layer', batch_norm=False)
            for i in range(1, hp['depth'] - 1):
                layer = _dense_layer(layer, (hidden_size, hidden_size), dropout_keep_prob, 'layer_' + str(i))
            logits = _dense_layer(layer, (hidden_size, hp['output_size']), 1., 'output_layer', activation=None, init=utils.linear_xavier_avg)
    pred = tf.reduce_sum(bucket_means * tf.nn.softmax(logits), axis=1)
    return pred, logits


def _embeddings(X, discrete_features, hp):
    with tf.variable_scope('embeddings'):
        features = []
        for feature_idx, values in discrete_features:
            features.append(Embedding(len(values), hp['embedding_dim'], embeddings_initializer='glorot_normal')(X[:, feature_idx]))
        for feature_idx in range(X.shape[-1].value):
            if len(discrete_features) == 0 or feature_idx not in list(zip(*discrete_features))[0]:
                features.append(tf.reshape(X[:, feature_idx], [-1, 1]))
        return tf.concat(features, axis=1)


def build_graph(n_input, discrete_features, hp, bucket_means, summarize=True, name=''):
    """ Build Tensorflow training, validation and testing graph """
    with tf.variable_scope(name):
        # Define inputs
        with tf.variable_scope('inputs'):
            lr = tf.placeholder(tf.float32, [], name='learning_rate')
            dropout_keep_prob = tf.placeholder_with_default(1., [], name='dropout_keep_prob')
            l2_reg = tf.placeholder(tf.float32, [], name='l2_regularization')
            X = tf.placeholder(tf.float32, [None, n_input], name='X')
            labels = tf.placeholder(tf.int32, [None], name='labels')
            y = tf.placeholder(tf.float32, [None], name='y')
            train = tf.placeholder_with_default(False, [], name='training')

        weigths_collection = 'weights' if name is None else name + '_weigths'
        X_embedded = _embeddings(X, discrete_features, hp) if 'embedding_dim' in hp else X
        bucket_means = tf.constant(bucket_means, dtype=tf.float32, name='buckets_means')
        pred, logits = _build_dnn(X_embedded, hp, bucket_means, dropout_keep_prob, summarize=summarize, training=train, weights_col=weigths_collection)
        tf.summary.histogram('pred', pred, collections=['extended_summary'])
        rmse = tf.sqrt(tf.losses.mean_squared_error(y, pred), name='rmse')

        # Define loss function (cross entropy and L2 regularization) and optimizer
        with tf.variable_scope('L2_regularization'):
            L2 = l2_reg * tf.add_n([tf.nn.l2_loss(w) for w in tf.get_collection(weigths_collection)])
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) + L2
        if 'warm_resart_lr' in hp:
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr, use_nesterov=True, momentum=hp['momentum'])
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        grads_and_vars = optimizer.compute_gradients(loss)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            apply_grad = optimizer.apply_gradients(grads_and_vars)
        train_op = tf.tuple([rmse, loss], control_inputs=[apply_grad], name='train_ops')

        # Variable initialization operation
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), name='init_op')
        return pred, rmse, loss, train_op, (X, y, labels, lr, dropout_keep_prob, l2_reg, train), init_op


def _placeholders_feed(placeholders, X, y=None, labels=None, lr=None, dropout_keep_prob=None, l2_reg=None, training=None):
    feed_dict = {}
    for (X_ph, y_ph, labels_ph, lr_ph, dropout_keep_prob_ph, l2_reg_ph, training_ph) in placeholders:
        def _add_placeholder(ph, value):
            if value is not None:
                feed_dict[ph] = value
        _add_placeholder(X_ph, X)
        _add_placeholder(y_ph, y)
        _add_placeholder(labels_ph, labels)
        _add_placeholder(lr_ph, lr)
        _add_placeholder(dropout_keep_prob_ph, dropout_keep_prob)
        _add_placeholder(l2_reg_ph, l2_reg)
        _add_placeholder(training_ph, training)
    return feed_dict


def train(models, dataset, hp, save_dir, testset):
    # Unpack parameters and tensors
    train_data, valid_data, train_targets, valid_targets, train_labels, _ = dataset
    preds, rmses, losses, train_ops, placeholders, init_ops = zip(*models)
    saver = tf.train.Saver()
    wr_hp = hp.get('warm_resart_lr')

    # Start tensorflow session
    with tf.Session(config=utils.tf_config(ALLOW_GPU_MEM_GROWTH)) as sess:
        sess.run(init_ops)

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
                batch_rmses, batch_losses = zip(*sess.run(train_ops, feed_dict=_placeholders_feed(placeholders,
                                                                                                  X=train_data[range_min:range_max],
                                                                                                  y=train_targets[range_min:range_max],
                                                                                                  labels=train_labels[range_min:range_max],
                                                                                                  lr=learning_rate,
                                                                                                  dropout_keep_prob=hp['dropout_keep_prob'],
                                                                                                  l2_reg=hp['l2_regularization'],
                                                                                                  training=True)))
                if wr_hp is not None:
                    learning_rate, new_cycle = utils.warm_restart(epoch + step / batch_per_epoch, t_0=wr_hp['initial_cycle_length'],
                                                                  max_lr=hp['lr'], min_lr=wr_hp['minimal_lr'], t_mult=wr_hp['lr_cycle_growth'])
                    if new_cycle:
                        print('New learning rate cycle.')
                        epochs_since_improvement = 0  # We reset early stoping for each lr cycles
                        cycle += 1
                mean_rmse += (range_max - range_min) * np.mean(batch_rmses) / len(train_data)
                mean_loss += (range_max - range_min) * np.mean(batch_losses) / len(train_data)
                mean_lr += (range_max - range_min) * learning_rate / len(train_data)

            # Evaluate model and display progress
            utils.add_summary_values(summary_writer, global_step=epoch, mean_rmse=mean_rmse, mean_loss=mean_loss, mean_lr=mean_lr)
            if epoch % EVALUATE_PERIOD == 0:
                # Process RMSE on validation set
                if (epoch // EVALUATE_PERIOD) % EXTENDED_SUMMARY_EVAL_PERIOD == 0:
                    *valid_rmses, summary = sess.run([*rmses, extended_summary_op], feed_dict=_placeholders_feed(placeholders, X=valid_data, y=valid_targets))
                    summary_writer.add_summary(summary, epoch)
                else:
                    valid_rmses = sess.run(rmses, feed_dict=_placeholders_feed(placeholders, X=valid_data, y=valid_targets))
                valid_rmse = np.mean(valid_rmses)
                utils.add_summary_values(summary_writer, global_step=epoch, valid_rmse=valid_rmse)
                # Save snapshot if validation RMSE is the best encountered so far
                if best_rmse > valid_rmse:
                    print('Best valid_rmse encountered so far, saving model...')
                    best_rmse = valid_rmse
                    epochs_since_improvement = 0
                    best_saveddir = os.path.join(save_dir, 'best_snapshot/')
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
                predictions.extend(np.mean(sess.run(preds, feed_dict=_placeholders_feed(placeholders, X=batch_X)), axis=0))
            return predictions

        print("Training done, best_rmse=%.6f" % best_rmse)
        ensemble_rmse = float('inf')
        if wr_hp is not None:
            # Apply last cycle snapshots on test and validation set and average softmax layers for prediction (snaphot ensembling)
            predictions, valid_preds = [], []
            for snapshot in range(wr_hp['keep_best_snapshot']):
                print('Applying snapshot #' + str(snapshot))
                saver.restore(sess, os.path.join(save_dir, str(snapshot)) + '/')
                valid_preds.append(np.mean(sess.run(preds, feed_dict=_placeholders_feed(placeholders, X=valid_data)), axis=0))
                predictions.append(_predict_testset())

            predictions = np.mean(predictions, axis=0)
            ensemble_rmse = np.sqrt(np.mean((np.mean(valid_preds, axis=0) - valid_targets) ** 2))
            print('Ensemble valid_rmse=' + str(ensemble_rmse))
        if ensemble_rmse > best_rmse:
            # Restore best model snapshot and apply prediction
            print("Snaphsot ensembling isn't enabled or didn't performed better than best rsmse encountered during training")
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
    features_len, discrete_feats, (test_ids, testset), dataset, bucket_means = feature_engineering.load_data(dataset_dir,
                                                                                                             'train.csv',
                                                                                                             'test.csv',
                                                                                                             VALID_SIZE,
                                                                                                             hyperparameters['output_size'],
                                                                                                             embed_discrete_features=(
                                                                                                                 'embedding_dim' in hyperparameters),
                                                                                                             max_distinct_values=hyperparameters.get(
                                                                                                                 'max_embedding_values'),
                                                                                                             cache_read_only=tf.flags.FLAGS.floyd_job)

    # Build models
    models = [build_graph(features_len, discrete_feats, hyperparameters, bucket_means, name='model_' + str(i)) for i in range(hyperparameters['n_models'])]

    # Train model
    print('Model built, starting training.')
    _, test_preds = train(models, dataset, hyperparameters, save_dir, testset)

    # Save predictions to csv file for Kaggle submission
    test_preds = np.int32(np.round(np.exp(test_preds))) - 1
    pd.DataFrame(np.column_stack([test_ids, test_preds]), columns=['id', 'trip_duration']).to_csv(os.path.join(save_dir, 'preds.csv'), index=False)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)  # Set log level to debug
    tf.flags.DEFINE_bool('floyd-job', False, 'Change working directories for training on Floyd.')
    tf.app.run()
