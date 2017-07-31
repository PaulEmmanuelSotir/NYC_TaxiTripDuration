#!/usr/bin/python
# -*- coding: utf-8 -*-
""" NYC Taxi Trip Duration - Kaggle competion
Note that the code could probably be greatly simplified using tf.train.Supervisor and tf.contrib.learn.dnnregressor,
but we prefer to define model by hand here to learn more about tensorflow python API.

TODO:
    * try batch normalization
    * try to train deeper models on more data

.. See https://github.com/PaulEmmanuelSotir/NYC_TaxiTripDuration and https://www.floydhub.com/paulemmanuel/projects/nyc_taxi_trip_duration
"""
import math
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

__all__ = ['load_data', 'build_model', 'train']

DEFAULT_HYPERPARAMETERS = {'lr': 0.0014642413409643805, 'batch_size': 256, 'weight_std_dev': 0.09628399771274719, 'hidden_size': 512, 'dropout_keep_prob': 0.822902075856152}
TRAINING_EPOCHS = 200
DISPLAY_STEP_PREDIOD = 1
ALLOW_GPU_MEM_GROWTH = True
TEST_SIZE = 0.1
RANDOM_STATE = 100 # Random state for train_test_split

def _haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def load_data(train_path, test_path):
    trainset = pd.read_csv(train_path)
    predset = pd.read_csv(test_path)
    # Remove outliers (huge trip durations)
    q = trainset.trip_duration.quantile(0.998)
    trainset = trainset[trainset.trip_duration < q]
    # Parse and vectorize dates
    def _preprocess_date(dataset, field, vectorize=True):
        dataset[field] = pd.to_datetime(dataset[field])
        if vectorize:
            dataset[field + '_hour'] = dataset[field].dt.hour
            dataset[field + '_min'] = dataset[field].dt.minute
            dataset[field + '_weekday'] = dataset[field].dt.weekday
            dataset[field + '_day'] = dataset[field].dt.day
    _preprocess_date(trainset, 'pickup_datetime')
    _preprocess_date(predset, 'pickup_datetime')
    _preprocess_date(trainset, 'dropoff_datetime', vectorize=False)
    # Vectorize flags
    trainset['store_and_fwd_flag'], _ = pd.factorize(trainset['store_and_fwd_flag'])
    predset['store_and_fwd_flag'], _ = pd.factorize(predset['store_and_fwd_flag'])
    # Process harversine distance from longitudes and latitudes
    trainset['radial_distance'] = _haversine_np(trainset.pickup_longitude, trainset.pickup_latitude, trainset.dropoff_longitude, trainset.dropoff_latitude)
    predset['radial_distance'] = _haversine_np(predset.pickup_longitude, predset.pickup_latitude, trainset.dropoff_longitude, predset.dropoff_latitude)
    # Transform target trip durations to log(trip durations) (permits to get a gaussian distribution of trip_durations, see data exploration notebook)
    trainset['trip_duration'] = np.log(trainset['trip_duration'] + 1)
    # Remove unused columns and split input feature columns from target column
    features = [key for key in trainset.keys().intersection(predset.keys()) if key != 'id' and key != 'pickup_datetime']
    targets, data = trainset['trip_duration'].get_values().reshape([-1, 1]), trainset[features].get_values()
    return features, predset, train_test_split(data, targets, test_size=TEST_SIZE, random_state=RANDOM_STATE)

def _mlp(_X, _weights, _biases, dropout_keep_prob):
    layer1 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1'])), dropout_keep_prob)
    layer2 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer1, _weights['w2']), _biases['b2'])), dropout_keep_prob)
    layer3 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer2, _weights['w3']), _biases['b3'])), dropout_keep_prob)
    layer4 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer3, _weights['w4']), _biases['b4'])), dropout_keep_prob)
    out = tf.add(tf.matmul(layer4, _weights['w_out']), _biases['b_out'])
    return out

def build_model(n_input, hidden_size, weight_std_dev, learning_rate):
    # Define placeholders
    X = tf.placeholder(tf.float32, [None, n_input], name='X')
    y = tf.placeholder(tf.float32, [None, 1], name='y')
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
    # Define MLP weights and biases variables
    with tf.name_scope('mlp'):
        n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4 = (hidden_size, hidden_size, hidden_size, hidden_size)
        weights = {'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=weight_std_dev), name='w1'),
                   'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=weight_std_dev), name='w2'),
                   'w3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=weight_std_dev), name='w3'),
                   'w4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], stddev=weight_std_dev), name='w4'),
                   'w_out': tf.Variable(tf.random_normal([n_hidden_4, 1], stddev=weight_std_dev), name='w_out')}
        biases = {'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
                  'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
                  'b3': tf.Variable(tf.random_normal([n_hidden_3]), name='b3'),
                  'b4': tf.Variable(tf.random_normal([n_hidden_4]), name='b4'),
                  'b_out': tf.Variable(tf.random_normal([1]), name='b_out')}
        # Build fully connected layers
        pred = _mlp(X, weights, biases, dropout_keep_prob)
    # Add variables to summary
    for v in [*biases.values(), *weights.values()]:
        tf.summary.histogram(v.name, v)
    # Define loss and optimizer
    loss = tf.losses.mean_squared_error(y, pred) # TODO: try NLL/cross entropy instead (with and without log pre-processing to see the effect of gaussian data distribution)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    tf.summary.scalar('mse_loss', loss)
    # Create model saver
    saver = tf.train.Saver()
    # Variable initialization operation
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    return loss, optimizer, (X, y, dropout_keep_prob), saver, init_op

def train(model, dataset, epochs, hyperparameters, save_dir=None):
    # Unpack parameters
    train_data, test_data, train_targets, test_targets = dataset
    loss, optimizer, placeholders, saver, init_op = model
    X, y, dropout_keep_prob = placeholders
    # Start tensorlfow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = ALLOW_GPU_MEM_GROWTH
    with tf.Session(config=config) as sess:
        sess.run(init_op)

        # Create summary utils
        if save_dir is not None:
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(save_dir, sess.graph)

        # Training loop
        for epoch in range(epochs):
            #avg_loss = 0. # TODO: remove it
            batch_count = len(train_data) // hyperparameters['batch_size']
            for _ in range(batch_count):
                indices = np.random.randint(len(train_data), size=hyperparameters['batch_size'])
                batch_xs = train_data[indices, :]
                batch_ys = train_targets[indices, :]
                # Fit using batched data
                sess.run(optimizer, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob: hyperparameters['dropout_keep_prob']})
                # TODO: find a cheap way to get average loss
            # Display progress
            if epoch % DISPLAY_STEP_PREDIOD == 0:
                if save_dir is not None:
                    summary, last_loss = sess.run([summary_op, loss], feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob: 1.})
                    summary_writer.add_summary(summary, epoch)
                else:
                    last_loss = sess.run(loss, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob: 1.})
                print("Epoch=%03d/%03d, last_loss=%.8f (rmse=%.6f)" % (epoch, epochs, last_loss, math.sqrt(last_loss)))
        # Calculate test MSE
        print("Training done, testing...")
        test_mse = sess.run(loss, feed_dict={X: test_data, y: test_targets, dropout_keep_prob:1.})
        print("test_mse=%.8f (rmse=%.4f)" % (test_mse, math.sqrt(test_mse)))
        # Save model
        if save_dir is not None:
            print('Saving model...')
            saver.save(sess, save_dir, global_step=epoch)
        return test_mse

def main():
    parser = argparse.ArgumentParser(description='Trains NYC Taxi trip duration fully connected neural network model for Kaggle competition submission.')
    parser.add_argument('--floyd-job', action='store_true', help='Change working directories for training on Floyd service')
    args = parser.parse_args()
    save_dir = '/output/model/' if args.floyd_job else './model/'
    train_set_path = '/input/train.csv' if args.floyd_job else './NYC_taxi_data_2016/train.csv'
    pred_set_path = '/input/test.csv' if args.floyd_job else './NYC_taxi_data_2016/test.csv'

    hyperparameters = DEFAULT_HYPERPARAMETERS

    # Parse and preprocess data
    features, predset, dataset = load_data(train_set_path, pred_set_path)

    # Build model
    model = build_model(len(features), hyperparameters['hidden_size'], hyperparameters['weight_std_dev'], hyperparameters['lr'])

    # Training model
    print('Model built, starting training.')
    train(model, dataset, TRAINING_EPOCHS, hyperparameters, save_dir)

if __name__ == '__main__':
    main()
