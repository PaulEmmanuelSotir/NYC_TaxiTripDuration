#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Hyperparameter optimization of DNN model
Uses hyperopt module to search for optimal DNN hyper parameters.

.. See https://github.com/PaulEmmanuelSotir/NYC_TaxiTripDuration
"""
from sklearn.utils import resample
import tensorflow as tf
import hyperopt as ho
import pandas as pd
import numpy as np
import argparse
import time
import math
import os

import utils
import nyc_dnn

#HP_SPACE = {'epochs': 200,
#            'early_stopping': 20,
#            'lr': ho.hp.loguniform('lr', math.log(2e-5), math.log(2e-3)),
#            'opt': ho.hp.choice('opt', [{'algo': tf.train.AdamOptimizer},
#                                        {'algo': tf.train.RMSPropOptimizer},
#                                        {'algo': tf.train.GradientDescentOptimizer, 'lr_decay': ho.hp.uniform('lr_decay', 0.5, 1.)},
#                                        {'algo': tf.train.MomentumOptimizer, 'lr_decay': ho.hp.uniform('lr_decay', 0.5, 1.), 'm': ho.hp.uniform('m', -0.05, .5)}]),
#            'depth': ho.hp.choice('depth', [4, 5, 6, 7, 8, 9]),
#            'batch_size': ho.hp.choice('batch_size', [512, 1024, 2048, 4096, 8192]),
#            'hidden_size': ho.hp.choice('hidden_size', [256, 512, 1024]),
#            'dropout_keep_prob': ho.hp.uniform('dropout_keep_prob', 0.75, 0.9),
#            'max_norm_threshold': ho.hp.uniform('max_norm_threshold', [0.8, 2.]),
#            'duration_std_margin': ho.hp.choice('duration_std_margin', [5, 6]),
#            'duration_resolution': ho.hp.choice('duration_resolution', [1, 128, 256, 512])}

# Hyperparameter optimization space and algorithm
MAX_EVALS = 35
SUB_TRAINSET_SIZE = 1.
ALLOW_GPU_MEM_GROWTH = True
OPT_ALGO = ho.tpe.suggest
HP_SPACE = {'epochs': 200,
            'early_stopping': 20,
            'lr': ho.hp.loguniform('lr', math.log(2e-5), math.log(2e-3)),
            'opt': {'algo': tf.train.AdamOptimizer},
            'depth': ho.hp.choice('depth', [4, 5]),
            'batch_size': ho.hp.choice('batch_size', [512, 1024, 2048, 4096, 8192]),
            'hidden_size': 512,
            'dropout_keep_prob': ho.hp.uniform('dropout_keep_prob', 0.75, 0.9),
            'max_norm_threshold': ho.hp.uniform('max_norm_threshold', 0.8, 2.),
            'duration_std_margin': 5,
            'output_size': 1}

def main():
    # Define working directories
    save_dir = '/output/models/hyperopt_models/' if tf.flags.FLAGS.floyd_job else './models/hyperopt_models/'
    dataset_dir = '/input/' if tf.flags.FLAGS.floyd_job else './NYC_taxi_data_2016/'

    # Load data and sample a subset of trainset
    knn_files = ('knn_train_features.npz', 'knn_test_features.npz', 'knn_pred_features.npz')
    features_len, (pred_ids, predset), dataset, (target_std, target_mean) = nyc_dnn.load_data(dataset_dir, 'train.csv', 'test.csv', *knn_files)
    train_data, test_data, train_targets, test_targets = dataset
    train_data, train_targets = resample(train_data, train_targets, replace=False, n_samples=int(SUB_TRAINSET_SIZE * len(train_data)))
    dataset = (train_data, test_data, train_targets, test_targets)


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
        model = nyc_dnn.build_model(features_len, hyperparameters, target_std, target_mean, summarize_parameters=False)
        # Train model
        test_rmse, predictions = nyc_dnn.train(model, dataset, hyperparameters, model_save_dir, predset)
        # Save predictions to csv file for Kaggle submission
        predictions = np.int32(np.round(np.exp(predictions))) - 1
        pd.DataFrame(np.column_stack([pred_ids, predictions]), columns=['id', 'trip_duration']).to_csv(os.path.join(save_dir, str(eval_count), 'preds.csv'), index=False)
        return {'loss': test_rmse, # TODO: put last batch average loss here?
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
    print(best_parameters) # TODO: translate ho.hp.choice hyperarameters from index to their actual value

if __name__ == '__main__':
    tf.flags.DEFINE_bool('floyd-job', False, 'Change working directories for training on Floyd.')
    tf.app.run()
