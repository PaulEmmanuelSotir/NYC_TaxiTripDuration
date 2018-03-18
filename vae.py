#!/usr/bin/python
# -*- coding: utf-8 -*-
""" VAE
Tensorflow implementation of a variationnal autencoder
"""
import tensorflow as tf
import numpy as np
import utils


class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE)
    Encoder and decoder networks are composed of fully connected layers.

    See "Auto-Encoding Variational Bayes" (https://arxiv.org/abs/1312.6114) for more details.
    """

    def __init__(self, min_x, max_x, lr, batch_size, n_inputs, n_z, decoder_layers, encoder_layers, clipping_threshold=None,
                 batch_norm=False, keep_prob=1., momentum=0.9, activation_fn=tf.nn.relu, init=utils.relu_xavier_avg):
        self.clipping_threshold = clipping_threshold
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.activation_fn = activation_fn
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.keep_prob = keep_prob
        self.n_inputs = n_inputs
        self.momentum = momentum
        self.init = init
        self.n_z = n_z
        self.lr = lr
        range_x = np.asarray(np.asarray(max_x) - np.array(min_x))
        range_x[range_x == 0] = 1.  # Make sure we don't have 0 ranges
        self.range_x = tf.constant(range_x, tf.float32, name='range_x')
        self.min_x = tf.constant(min_x, tf.float32, name='min_x')

        with tf.variable_scope('VariationalAutoEncoder'):
            # Define input placeholders
            self.x = tf.placeholder(tf.float32, [None, n_inputs], name='X')
            self.norm_x = tf.clip_by_value((self.x - self.min_x) / self.range_x, 0., 1.)
            self.training = tf.placeholder_with_default(False, [], name='training')  # Needed for batch normalization
            # Define encoder and decoder networks
            self._encoder()
            self._decoder()
            # Define loss function based on variational upper-bound and corresponding optimizer
            self._loss_optimizer()
            # Define tensorflow variables intialization op
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), name='init_op')

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _dense_layer(self, x, shape, batch_norm, init, activation_fn=None, keep_prob=1., name=None):
        with tf.variable_scope(name):
            weights = tf.get_variable(initializer=init(shape), name='w',
                                      collections=['weights', tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES])
            bias = tf.get_variable(initializer=tf.truncated_normal([shape[1]]) if shape[1] > 1 else 0., name='b')
            logits = tf.add(tf.matmul(x, weights), bias)
            if batch_norm:
                logits = tf.layers.batch_normalization(logits, training=self.training, name='batch_norm')
            dense = activation_fn(logits) if activation_fn is not None else logits
            dense = tf.nn.dropout(dense, keep_prob)
            image = tf.reshape(weights, [1, weights.shape[0].value, weights.shape[1].value, 1])
            tf.summary.image('weights', image, collections=['extended_summary'])
            tf.summary.histogram('weights_histogram', weights, collections=['extended_summary'])
            tf.summary.histogram('bias', bias, collections=['extended_summary'])
        return dense

    def _encoder(self):
        n_hidden = self.encoder_layers
        with tf.variable_scope('encoder'):
            # Define encoder network which maps inputs to a normal distribution in latent space
            layer = self._dense_layer(self.norm_x, (self.n_inputs, n_hidden[0]), False, self.init, self.activation_fn, self.keep_prob, name='input_layer')
            for i, shape in
                layer = self._dense_layer(layer, shape, self.batch_norm, self.init, self.activation_fn, self.keep_prob, name='layer_' + str(i + 1))
            # TODO: see whether if using tf.split on a single output layer yields to better performances
            self.z_mean = self._dense_layer(layer, (n_hidden[-1], self.n_z), self.batch_norm, utils.linear_xavier_avg, name='z_mean')
            self.z_sq_log_stddev = self._dense_layer(layer, (n_hidden[-1], self.n_z), self.batch_norm, utils.linear_xavier_avg, name='z_sq_log_stddev')
            # Draw z from a gaussian distribution parameterized by encoder network outputs
            self.z = self.z_mean + tf.sqrt(tf.exp(self.z_sq_log_stddev)) * tf.random_normal(tf.shape(self.z_mean))

    def _decoder(self):
        n_hidden = self.decoder_layers
        with tf.variable_scope('decoder'):
            # Define decoder network which maps latent vectors to generated data vectors
            layer = self._dense_layer(self.z, (self.n_z, n_hidden[0]), False, self.init, self.activation_fn, self.keep_prob, name='input_layer')
            for i, shape in enumerate(zip(n_hidden[:-1], n_hidden[1:])):
                layer = self._dense_layer(layer, shape, self.batch_norm, self.init, self.activation_fn, self.keep_prob, name='layer_' + str(i + 1))
            self.x_decoded_norm = self._dense_layer(layer, (n_hidden[-1], self.n_inputs), self.batch_norm, utils.linear_xavier_avg, tf.nn.sigmoid, name='out')
            self.x_decoded = self.range_x * self.x_decoded_norm + self.min_x

    def _loss_optimizer(self):
        epsilon = 1e-7  # Avoids log(0)
        # Reconstruction loss is the negative log likelihood of the input under the reconstructed Bernoulli distribution induced by the decoder in the data space
        reconstr_loss = - self.n_inputs * tf.reduce_mean(self.norm_x * tf.log(epsilon + self.x_decoded_norm) +
                                                         (1 - self.norm_x) * tf.log(epsilon + 1 - self.x_decoded_norm), axis=1)
        reconstr_loss = tf.tuple([reconstr_loss], control_inputs=[tf.assert_non_negative(1 - self.norm_x),
                                                                  tf.verify_tensor_all_finite(self.z_mean, 'oups'),
                                                                  tf.verify_tensor_all_finite(self.norm_x, 'oups2'),
                                                                  tf.verify_tensor_all_finite(reconstr_loss, 'oups3')])
        # Latent loss is defined as the Kullback Leibler divergence between the distribution in latent space induced by the encoder and standard Gaussian prior
        latent_loss = - 0.5 * self.n_z * tf.reduce_mean(1 + self.z_sq_log_stddev - tf.square(self.z_mean) - tf.exp(self.z_sq_log_stddev), axis=1)
        self.loss = tf.reduce_mean(reconstr_loss + latent_loss, name='loss') / (self.n_inputs + self.n_z)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum, use_nesterov=True)
        gradients = optimizer.compute_gradients(self.loss)
        if self.clipping_threshold is not None:
            gradients = [(tf.clip_by_norm(grad, self.clipping_threshold), var) for grad, var in gradients]
        self.optimizer = optimizer.apply_gradients(gradients)

    def fit_batch(self, X):
        """Train model on minibatch data """
        _, loss = self.sess.run((self.optimizer, self.loss), feed_dict={self.x: X, self.training: True})
        return loss

    def encode(self, X):
        """ Apply encoder to X return latent mean and squared log std dev of gaussian distribution """
        return self.sess.run((self.z_mean, self.z_sq_log_stddev), feed_dict={self.x: X})

    def decode(self, Z=None):
        """ Decode given latent vector. If Z is None, Z is drawn from standard gaussian distribution """
        if Z is None:
            Z = np.random.normal(size=self.n_z)
        return self.sess.run(self.x_decoded, feed_dict={self.z: Z})

    def reconstruct(self, X):
        """ Use VAE to encode then decode given data """
        return self.sess.run(self.x_decoded, feed_dict={self.x: X})
