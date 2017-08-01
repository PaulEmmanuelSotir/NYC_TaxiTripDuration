#!/usr/bin/python
# -*- coding: utf-8 -*-
""" NYC Taxi Trip Duration - Kaggle competion
Note that the code could probably be greatly simplified using tf.train.Supervisor and tf.contrib.learn.dnnregressor,
but we prefer to define model by hand here to learn more about tensorflow python API.

TODO:
    * try batch normalization
    * try to train deeper models on more data
    * try to dicretize trip_duration and use softmax layer for classification instead of a regression with linear output
    * determine whether if cross validation could improve accuracy

.. See https://github.com/PaulEmmanuelSotir/NYC_TaxiTripDuration
"""
import os
import math
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

__all__ = ['load_data', 'build_model', 'train']

DEFAULT_HYPERPARAMETERS = {'lr': 0.00003869398376828445, 'depth': 9, 'dropout_keep_prob': 0.64893328427820518, 'hidden_size': 512, 'batch_size': 512, 'weight_std_dev': 0.10134450718453812}
TRAINING_EPOCHS = 250
DISPLAY_STEP_PREDIOD = 2
ALLOW_GPU_MEM_GROWTH = True
TEST_SIZE = 0.1
RANDOM_STATE = 100 # Random state for train_test_split

def _haversine_np(lon1, lat1, lon2, lat2):
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
    return features, (predset['id'], predset[features].get_values()), train_test_split(data, targets, test_size=TEST_SIZE, random_state=RANDOM_STATE)

def _mlp(X, weights, biases, dropout_keep_prob):
    layers = [tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(X, weights[0]), biases[0])), dropout_keep_prob)]
    for w, b in zip(weights[1:-1], biases[1:-1]):
        layers.append(tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layers[-1], w), b)), dropout_keep_prob))
    out = tf.add(tf.matmul(layers[-1], weights[-1]), biases[-1], name='output')
    return out
    
def build_model(n_input, depth, hidden_size, weight_std_dev, learning_rate):
    # Define placeholders
    X = tf.placeholder(tf.float32, [None, n_input], name='X')
    y = tf.placeholder(tf.float32, [None, 1], name='y')
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    with tf.name_scope('mlp'):
        # Define MLP weights and biases variables
        weights = [tf.Variable(tf.random_normal([n_input, hidden_size], stddev=weight_std_dev), name='w1')]
        biases = [tf.Variable(tf.random_normal([hidden_size]), name='b1')]
        for i in range(1, depth-1):
            weights.append(tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=weight_std_dev), name='w' + str(i)))
            biases.append(tf.Variable(tf.random_normal([hidden_size]), name='b' + str(i)))
        weights.append(tf.Variable(tf.random_normal([hidden_size, 1], stddev=weight_std_dev), name='w_out'))
        biases.append(tf.Variable(tf.random_normal([1]), name='b_out'))
        # Build fully connected layers
        pred = _mlp(X, weights, biases, dropout_keep_prob)
    # Add variables to summary
    for v in [*biases, *weights]:
        tf.summary.histogram(v.name, v)
    # Define loss and optimizer
    loss = tf.losses.mean_squared_error(y, pred) # TODO: try tf.losses.huber_loss instead
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    tf.summary.scalar('mse_loss', loss)
    # Create model saver
    saver = tf.train.Saver()
    # Variable initialization operation
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    return pred, loss, optimizer, (X, y, dropout_keep_prob), saver, init_op

def train(model, dataset, epochs, hyperparameters, save_dir=None, predset=None):
    # Unpack parameters
    train_data, test_data, train_targets, test_targets = dataset
    pred, loss, optimizer, placeholders, saver, init_op = model
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
                test_mse = sess.run(loss, feed_dict={X: test_data, y: test_targets, dropout_keep_prob: 1.})
                print("Epoch=%03d/%03d, last_loss=%.6f (rmse=%.6f), test_mse=%.6f (rmse=%.6f)" % (epoch, epochs, last_loss, math.sqrt(last_loss), test_mse, math.sqrt(test_mse)))
        # Calculate test MSE
        print("Training done, testing...")
        test_mse = sess.run(loss, feed_dict={X: test_data, y: test_targets, dropout_keep_prob: 1.})
        print("test_mse=%.8f (rmse=%.4f)" % (test_mse, math.sqrt(test_mse)))
        # Save model
        if save_dir is not None:
            print('Saving model...')
            saver.save(sess, save_dir, global_step=epoch)
        if predset is not None:
            print('Applying trained model on prediction set')
            predictions = []
            for batch in np.array_split(predset, len(predset) // hyperparameters['batch_size']):
                predictions.extend(sess.run(pred, feed_dict={X: batch, dropout_keep_prob: 1.}).flatten())
            return test_mse, predictions
        return test_mse

def main():
    # Parse cmd arguments
    parser = argparse.ArgumentParser(description='Trains NYC Taxi trip duration fully connected neural network model for Kaggle competition submission.')
    parser.add_argument('--floyd-job', action='store_true', help='Change working directories for training on Floyd service')
    args = parser.parse_args()
    save_dir = '/output/model/' if args.floyd_job else './model/'
    train_set_path = '/input/train.csv' if args.floyd_job else './NYC_taxi_data_2016/train.csv'
    pred_set_path = '/input/test.csv' if args.floyd_job else './NYC_taxi_data_2016/test.csv'

    # Parse and preprocess data
    features, (pred_ids, predset), dataset = load_data(train_set_path, pred_set_path)

    # Build model
    hyperparameters = DEFAULT_HYPERPARAMETERS
    model = build_model(len(features), hyperparameters['depth'], hyperparameters['hidden_size'], hyperparameters['weight_std_dev'], hyperparameters['lr'])

    # Train model
    print('Model built, starting training.')
    test_mse, predictions = train(model, dataset, TRAINING_EPOCHS, hyperparameters, save_dir, predset)

    # Save predictions to csv file for Kaggle submission
    predictions = np.int32(np.round(np.exp(predictions))) - 1
    predictions = [p if p < 5000 else 5000 for p in predictions] # TODO: temporary, correct explosing values problem
    pd.DataFrame(np.column_stack([pred_ids, predictions]), columns=['id', 'trip_duration']).to_csv(os.path.join(save_dir, 'preds.csv'), index=False)

if __name__ == '__main__':
    main()
