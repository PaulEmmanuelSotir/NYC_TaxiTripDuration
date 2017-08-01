#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Hyperparameter optimization of DNN model
Uses hyperopt module to search for optimal DNN hyper parameters.

TODO:
    * avoid writing to many tf summaries
    * handle early stoping and progressively increase dataset size (or/and NN depth) during hyperparameter optimization to accelerate optmization and allow more model evaluations

.. See https://github.com/PaulEmmanuelSotir/NYC_TaxiTripDuration
"""
from sklearn.utils import resample
import tensorflow as tf
import hyperopt as ho
import numpy as np
import argparse
import time
import math

import nyc_dnn

TRAINING_EPOCHS = 100
SUB_TRAINSET_SIZE = 0.1

# Hyperparameter optimization space and algorithm
MAX_EVALS = 100
OPT_ALGO = ho.tpe.suggest
HP_SPACE = {'lr': ho.hp.loguniform('lr', math.log(1e-6), math.log(1e-2)),
            'depth': ho.hp.quniform('depth', 6, 9, 1),
            'batch_size': ho.hp.choice('batch_size', [128, 256, 512]),
            'hidden_size': ho.hp.choice('hidden_size', [64, 128, 256, 512, 1024]),
            'weight_std_dev': ho.hp.normal('weight_std_dev', 0.1, 0.01),
            'dropout_keep_prob': ho.hp.normal('dropout_keep_prob', 0.7, 0.03)}

def _correct_hyperparameters(hyperparameters):
    hyperparameters['depth'] = int(hyperparameters['depth'])
    hyperparameters['weight_std_dev'] = np.clip(hyperparameters['weight_std_dev'], 1e-9, 1.)
    hyperparameters['dropout_keep_prob'] = np.clip(hyperparameters['dropout_keep_prob'], 0.05, 1.)

def main():
    # Parse cmd arguments
    parser = argparse.ArgumentParser(description='Trains NYC Taxi trip duration fully connected neural network model for Kaggle competition submission.')
    parser.add_argument('--floyd-job', action='store_true', help='Change working directories for training on Floyd service')
    args = parser.parse_args()
    save_dir = '/output/hyperopt_models/' if args.floyd_job else './hyperopt_models/'
    train_set_path = '/input/train.csv' if args.floyd_job else './NYC_taxi_data_2016/train.csv'
    pred_set_path = '/input/test.csv' if args.floyd_job else './NYC_taxi_data_2016/test.csv'

    # Parse, preprocess data and sample a subset of trainset
    features, predset, dataset = nyc_dnn.load_data(train_set_path, pred_set_path)
    train_data, test_data, train_targets, test_targets = dataset
    train_data, train_targets = resample(train_data, train_targets, replace=False, n_samples=int(SUB_TRAINSET_SIZE * len(train_data)))
    dataset = (train_data, test_data, train_targets, test_targets)

    # Define objective function optimized by hyperopt
    eval_count = 0
    def _objective(hyperparameters):
        _correct_hyperparameters(hyperparameters)
        nonlocal eval_count
        eval_count += 1
        print('\n\n' + '#' * 10 + ' TRYING HYPERPERPARAMETER SET #' + str(eval_count) + ' ' + '#' * 10)
        print(hyperparameters)
		# Reset default tensorflow graph
        tf.reset_default_graph()
        # Build model
        model = nyc_dnn.build_model(len(features), hyperparameters['depth'], hyperparameters['hidden_size'], hyperparameters['weight_std_dev'], hyperparameters['lr'])
        # Train model
        model_save_dir = save_dir + str(eval_count) + '/' # TODO: find a better way to do this (probably using hyperopt module features)
        test_mse = nyc_dnn.train(model, dataset, TRAINING_EPOCHS, hyperparameters, model_save_dir)
        return {'loss': test_mse, # TODO: put last batch average loss here?
                'true_loss': test_mse,
                'status': ho.STATUS_OK,
                'eval_time': time.time()}

    # Run optimization algorithm
    trials = ho.Trials()
    best_parameters = ho.fmin(_objective,
                              space=HP_SPACE,
                              algo=OPT_ALGO,
                              max_evals=MAX_EVALS,
                              trials=trials)
    _correct_hyperparameters(best_parameters)
    print('\n\n' + '#' * 20 + '  BEST HYPERPARAMETERS  ' + '#' * 20)
    print(best_parameters)

if __name__ == '__main__':
    main()
