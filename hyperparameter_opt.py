#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Hyperparameter optimization of DNN model
Uses hyperopt module to search for optimal DNN hyper parameters.
"""
import tensorflow as tf
import hyperopt as ho
import numpy as np
import time
import math

import nyc_dnn

# TODO: handle early stoping and progressively increase dataset size during optimization (or/and even NN depth) to accelerate optmization

# Define train/predict set paths and save directory
USE_FLOYD = True # TODO: take it as command arg
SAVE_DIR = '/output/hyperopt_models/' if USE_FLOYD else './hyperopt_models/'
TRAIN_SET = '/input/train.csv' if USE_FLOYD else './NYC_taxi_data_2016/train.csv'
PRED_SET = '/input/test.csv' if USE_FLOYD else './NYC_taxi_data_2016/test.csv'

# Hyperparameter optimization space and algorithm
TRAINING_EPOCHS = 30
MAX_EVALS = 100
OPT_ALGO = ho.tpe.suggest
HP_SPACE = {'lr': ho.hp.loguniform('lr', math.log(1e-5), math.log(1e-2)),
            'batch_size': ho.hp.quniform('batch_size', 32, 256, 1),
            'hidden_size': ho.hp.quniform('hidden_size', 128, 1024, 1),
            'weight_std_dev': ho.hp.normal('weight_std_dev', 0.1, 0.01), # TODO: clip normal to avoid invalid std dev
            'dropout_keep_prob': ho.hp.normal('dropout_keep_prob', 0.7, 0.05)} # TODO: clip normal to avoid invalid dropout prob

def main():
    # Parse and preprocess data
    features, predset, dataset = nyc_dnn.load_data(TRAIN_SET, PRED_SET)
    dataset = tuple(d[:20000] for d in dataset) # TODO: temporary

    # Define objective function optimized by hyperopt
    eval_count = 0
    def _objective(hyperparameters):
        nonlocal eval_count
        eval_count += 1
        hyperparameters['hidden_size'] = int(hyperparameters['hidden_size'])
        hyperparameters['batch_size'] = int(hyperparameters['batch_size'])
        print('\n\n' + '#' * 10 + ' TRYING HYPERPERPARAMETER SET #' + str(eval_count) + ' ' + '#' * 10)
        print(hyperparameters)
		# Reset default tensorflow graph
        tf.reset_default_graph()
        # Build model
        model = nyc_dnn.build_model(len(features), hyperparameters['hidden_size'], hyperparameters['weight_std_dev'], hyperparameters['lr'])
        # Train model
        model_save_dir = SAVE_DIR + str(eval_count) + '/' # TODO: find a better way to do this (probably using hyperopt possibilities)
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
    print('\n\n' + '#' * 20 + '  BEST HYPERPARAMETERS  ' + '#' * 20)
    print(best_parameters)

if __name__ == '__main__':
    main()
