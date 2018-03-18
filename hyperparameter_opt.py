#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Hyperparameter optimization of DNN model
Uses hyperopt module to search for optimal DNN hyper parameters.

Note: 'output_size' can't be changed during hyperparameter search as data preprocessing depends on this parameter.
      If you need to search over this parameter, make sure to re-execute 'feature_engineering.load_data' in '_objective' function (can be expensive).

.. See https://github.com/PaulEmmanuelSotir/NYC_TaxiTripDuration
"""
from sklearn.utils import resample
import tensorflow as tf
import hyperopt as ho
import pandas as pd
import numpy as np
import time
import math
import os

import utils
import nyc_dnn

# Full example of hyperparameter search space:
# HP_SPACE = {'epochs': 200,
#            'early_stopping': 20,
#            'lr': ho.hp.loguniform('lr', math.log(2e-5), math.log(2e-3)),
#            'momentum': ho.hp.uniform('momentum', 0.8, 0.95),
#            'depth': ho.hp.choice('depth', [4, 5, 6, 7, 8, 9]),
#            'hidden_size': ho.hp.choice('hidden_size', [256, 512, 1024]),
#            'batch_size': ho.hp.choice('batch_size', [256, 512, 1024, 2048]),
#            'dropout_keep_prob': ho.hp.uniform('dropout_keep_prob', 0.75, 0.9),
#            'l2_regularization': ho.hp.loguniform('l2_regularization', math.log(1e-6), math.log(1e-3)),
#            'output_size': ho.hp.choice('output_size', [1, 128, 256, 512])}

# Hyperparameter optimization space and algorithm
MAX_EVALS = 100
SUB_TRAINSET_SIZE = 1.
ALLOW_GPU_MEM_GROWTH = True
OPT_ALGO = ho.tpe.suggest
HP_SPACE = {'epochs': 50,
            'early_stopping': 10,
            'lr': ho.hp.loguniform('lr', math.log(2e-5), math.log(2e-3)),  # TODO: add tf.train.exponential_decay
            'momentum': ho.hp.uniform('momentum', 0.8, 0.95),
            'depth': 6,
            'embedding_dim': 8,
            'max_embedding_values': 64,
            'hidden_size': 512,
            'batch_size': 1024,
            'dropout_keep_prob': ho.hp.uniform('dropout_keep_prob', 0.7, 1.0),
            'l2_regularization': ho.hp.loguniform('l2_regularization', math.log(1e-7), math.log(1e-3)),
            'output_size': 512}


def main(_=None):
    # Define working directories
    source_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = '/output/' if tf.flags.FLAGS.floyd_job else os.path.join(source_dir, 'models/hyperopt_models/')
    dataset_dir = '/input/' if tf.flags.FLAGS.floyd_job else os.path.join(source_dir, 'NYC_taxi_data_2016/')

    # Load and preprocess data
    features_len, (test_ids, testset), dataset, bucket_means = nyc_dnn.feature_engineering.load_data(dataset_dir,
                                                                                                     'train.csv',
                                                                                                     'test.csv',
                                                                                                     nyc_dnn.VALID_SIZE,
                                                                                                     HP_SPACE['output_size'],
                                                                                                     tf.flags.FLAGS.floyd_job)

    # Reduce trainset size (sample a subset of trainset)
    train_data, test_data, train_targets, test_targets, train_labels, test_labels = dataset
    train_data, train_targets, train_labels = resample(train_data, train_targets, train_labels,
                                                       replace=False, n_samples=int(SUB_TRAINSET_SIZE * len(train_data)))
    dataset = (train_data, test_data, train_targets, test_targets, train_labels, test_labels)

    # Define the objective function optimized by hyperopt
    eval_count = 0

    def _objective(hyperparameters):
        nonlocal eval_count
        eval_count += 1
        model_save_dir = save_dir + str(eval_count) + '/'
        print('\n\n' + '#' * 10 + ' TRYING HYPERPERPARAMETER SET #' + str(eval_count) + ' ' + '#' * 10)
        print(hyperparameters)
        # Reset default tensorflow graph
        tf.reset_default_graph()
        # Write hyperparameters to summary as text (once per session/model)
        hp_string = tf.placeholder(tf.string, name='hp')
        hp_summary_op = tf.summary.text('hyperparameters', hp_string, collections=['per_session'])
        with tf.Session(config=utils.tf_config(ALLOW_GPU_MEM_GROWTH)) as sess:
            summary_writer = tf.summary.FileWriter(model_save_dir, sess.graph)
            summary = sess.run(hp_summary_op, feed_dict={hp_string: 'Trial #' + str(eval_count) + ' hyperparameters:\n' + str(hyperparameters)})
            summary_writer.add_summary(summary, 0)
        # Build model
        model = nyc_dnn.build_graph(features_len, hyperparameters, bucket_means, summarize=False)
        # Train model
        test_rmse, predictions = nyc_dnn.train(model, dataset, hyperparameters, model_save_dir, testset)
        # Save predictions to csv file for Kaggle submission
        predictions = np.int32(np.round(np.exp(predictions))) - 1
        pd.DataFrame(np.column_stack([test_ids, predictions]), columns=['id', 'trip_duration']
                     ).to_csv(os.path.join(save_dir, str(eval_count), 'preds.csv'), index=False)
        return {'loss': test_rmse,  # TODO: put last batch average loss here?
                'true_loss': test_rmse,
                'status': ho.STATUS_OK,
                'eval_time': time.time()}

    # Run optimization algorithm
    trials = ho.Trials()
    best_parameters = ho.fmin(_objective,
                              space=HP_SPACE,
                              algo=OPT_ALGO,
                              max_evals=MAX_EVALS,
                              trials=trials)
    print('\n\n' + '#' * 20 + '  BEST HYPERPARAMETERS  ' + '#' * 20)
    print(best_parameters)  # TODO: translate ho.hp.choice hyperarameters from index to their actual value


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)  # Set log level to debug
    tf.flags.DEFINE_bool('floyd-job', False, 'Change working directories for training on Floyd.')
    tf.app.run()
