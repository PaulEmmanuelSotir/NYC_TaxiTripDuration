#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Hyperparameter optimization of DNN model
Uses hyperopt module to search for optimal DNN hyper parameters.

TODO:
    * handle early stoping and progressively increase dataset size (or/and NN depth) during hyperparameter optimization to accelerate optmization and allow more model evaluations

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

import nyc_dnn

TRAINING_EPOCHS = 200
SUB_TRAINSET_SIZE = 1.

# Hyperparameter optimization space and algorithm
MAX_EVALS = 40
OPT_ALGO = ho.tpe.suggest
HP_SPACE = {'lr': ho.hp.loguniform('lr', math.log(1e-4), math.log(8e-3)),
            'lr_decay': ho.hp.uniform('lr_decay', 0.2, 1.),
            'activation': ho.hp.choice('activation', [tf.nn.tanh]),
            'batch_size': ho.hp.choice('batch_size', [1024, 2048, 2048]),
            'hidden_size': ho.hp.choice('hidden_size', [128, 256, 512]),
            'residual_blocks': ho.hp.choice('residual_blocks', [3, 4, 5, 6]),
            'l2_regularization': ho.hp.uniform('l2_regularization', 0, 0.05),
            'dropout_keep_prob': ho.hp.uniform('dropout_keep_prob', 0.7, 1.),
            'duration_std_margin': ho.hp.choice('duration_std_margin', [4, 5, 6]),
            'duration_resolution': ho.hp.choice('duration_resolution', [128, 256, 512])}

def main():
    # Parse cmd arguments
    parser = argparse.ArgumentParser(description='Trains NYC Taxi trip duration fully connected neural network model for Kaggle competition submission.')
    parser.add_argument('--floyd-job', action='store_true', help='Change working directories for training on Floyd service')
    args = parser.parse_args()
    save_dir = '/output/hyperopt_models/' if args.floyd_job else './hyperopt_models/'
    train_set_path = '/input/train.csv' if args.floyd_job else './NYC_taxi_data_2016/train.csv'
    pred_set_path = '/input/test.csv' if args.floyd_job else './NYC_taxi_data_2016/test.csv'

    # Load data and sample a subset of trainset
    features, (pred_ids, predset), dataset, (target_std, target_mean) = nyc_dnn.load_data(train_set_path, pred_set_path)
    train_data, test_data, train_targets, test_targets = dataset
    train_data, train_targets = resample(train_data, train_targets, replace=False, n_samples=int(SUB_TRAINSET_SIZE * len(train_data)))
    dataset = (train_data, test_data, train_targets, test_targets)

    # Define the objective function optimized by hyperopt
    eval_count = 0
    def _objective(hyperparameters):
        nonlocal eval_count # TODO: find a better way to do this (probably using hyperopt module features)
        eval_count += 1
        print('\n\n' + '#' * 10 + ' TRYING HYPERPERPARAMETER SET #' + str(eval_count) + ' ' + '#' * 10)
        print(hyperparameters)
		# Reset default tensorflow graph
        tf.reset_default_graph()
        # Build model
        model = nyc_dnn.build_model(len(features), hyperparameters, target_std, target_mean)
        # Train model
        model_save_dir = save_dir + str(eval_count) + '/'
        test_rmse, predictions = nyc_dnn.train(model, dataset, TRAINING_EPOCHS, hyperparameters, model_save_dir, predset)
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
    main()
