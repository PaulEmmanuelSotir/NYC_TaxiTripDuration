#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Utils

.. See https://github.com/PaulEmmanuelSotir/NYC_TaxiTripDuration
"""
import re
import os
import math
import shutil
import subprocess
import numpy as np
import tensorflow as tf

__all__ = ['tf_config', 'leaky_relu', 'tanh_xavier_avg', 'relu_xavier_avg', 'linear_xavier_avg', 'warm_restart',
           'add_summary_values', 'cd', 'floyd_run', 'floyd_stop', 'floyd_delete', 'get_model_from_floyd']

PREDS_FILE = 'preds.csv'

CMD_ENCODING = 'latin-1'
SCORES_FILE = 'scores.npz'
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

# Xavier initialization helpers
RELU_XAVIER_SCALE = 2.
TANH_XAVIER_SCALE = 4.
LINEAR_XAVIER_SCALE = 1.
relu_xavier_avg = tf.variance_scaling_initializer(RELU_XAVIER_SCALE, mode="fan_avg")
tanh_xavier_avg = tf.variance_scaling_initializer(TANH_XAVIER_SCALE, mode="fan_avg")
linear_xavier_avg = tf.variance_scaling_initializer(LINEAR_XAVIER_SCALE, mode="fan_avg")


def tf_config(allow_growth=True, **kwargs):
    config = tf.ConfigProto(**kwargs)
    config.gpu_options.allow_growth = allow_growth
    return config


def leaky_relu(x, leak=0.2, name='leaky_relu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def _cosine_annealing(x):
    return (np.cos(np.pi * x) + 1.) / 2.


def _log_cosine_annealing(x):
    log = np.log((np.exp(2) - np.exp(0)) * x + np.exp(0)) / 2.
    return (np.cos(np.pi * log) + 1.) / 2.


def warm_restart(epoch, t_0, max_lr, min_lr=1e-8, t_mult=2, annealing_fn=_log_cosine_annealing):
    """ Stochastic gradient descent with warm restarts of learning rate (see https://arxiv.org/pdf/1608.03983.pdf) """
    def _cycle_length(c): return t_0 * t_mult ** c
    cycle = int(np.floor(np.log(1 - epoch / t_0 * (1 - t_mult)) / np.log(t_mult)))
    cycle_begining = np.sum([_cycle_length(c) for c in range(0, cycle)]) if cycle > 0 else 0.
    x = (epoch - cycle_begining) / _cycle_length(cycle)
    lr = min_lr + (max_lr - min_lr) * annealing_fn(x)
    return lr, x == 0.


def add_summary_values(summary_writer, global_step=None, **values):
    if len(values) > 0:
        summary = tf.Summary()
        for name, value in values.items():
            summary.value.add(tag=name, simple_value=value)
        summary_writer.add_summary(summary, global_step=global_step)


class cd:
    """Context manager for changing the current working directory from https://stackoverflow.com/a/13197763/5323273"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def floyd_run(python_cmd, dataset=None, env='tensorflow-1.2', gpu=True):
    # Run floyd command to trigger job
    run_cmd = 'floyd run' + (' --data ' + dataset if dataset is not None else ' ') + ' --env ' + env + (' --gpu ' if gpu else ' ') + '"' + python_cmd + '"'
    print('>' + run_cmd + '\n')
    try:
        p = subprocess.run(run_cmd, stdout=subprocess.PIPE, shell=True, check=True, stderr=subprocess.STDOUT, cwd=SOURCE_DIR)
        out = p.stdout.strip().decode(CMD_ENCODING)
        matches = re.search('JOB NAME\s+-+\s+(.*/)+([0-9]+)', out)
        return int(matches.group(2))  # Return Floyd job number
    except subprocess.CalledProcessError as e:
        print('FAILED FLOYD COMMAND:\n> ' + e.cmd + '\nSTDOUT:\n' + e.output.decode(CMD_ENCODING))
    return -1


def floyd_stop(floyd_project, floyd_job, print_status=True):
    if print_status:
        print('Stoping Floyd job...')
    stop_cmd = 'floyd stop ' + os.path.join(floyd_project, str(floyd_job))
    try:
        subprocess.run(stop_cmd, stdout=subprocess.PIPE, shell=True, check=True, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        if print_status:
            print('FAILED TO STOP FLOYD JOB:\n> ' + e.cmd + '\nSTDOUT:\n' + e.output.decode(CMD_ENCODING))
    return False


def floyd_delete(floyd_project, floyd_job, stop_if_running=False):
    print('Deleting Floyd job...')
    delete_cmd = 'floyd delete ' + os.path.join(floyd_project, str(floyd_job)) + ' -y'
    if stop_if_running:
        floyd_stop(floyd_project, floyd_job, print_status=False)
    try:
        subprocess.run(delete_cmd, stdout=subprocess.PIPE, shell=True, check=True, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        print('FAILED TO DELETE FLOYD JOB:\n> ' + e.cmd + '\nSTDOUT:\n' + e.output.decode(CMD_ENCODING))
    return False


def get_model_from_floyd(floyd_project, floyd_job, models_dir, score=None, hyperparameters=None, delete_job=False):
    # Create scores file if it doesn't exist yet
    scores_path = os.path.join(models_dir, SCORES_FILE)
    if not os.path.isfile(scores_path):
        print('Creating a new score file...')
        with open(scores_path, 'wb') as file:
            np.save(file, np.array([]))
    scores = np.load(scores_path)

    # Verify that we didn't already imported this job score
    already_done = [s for s in scores if s['floyd_job'] == floyd_job]
    if len(already_done) > 0:
        print('Already imported this Floyd job score.')
        return already_done[0]

    # Download output directory
    print('downloading job output...')
    output_dir = os.path.join(models_dir, 'job_' + str(floyd_job))
    os.makedirs(output_dir)
    clone_cmd = 'floyd data clone ' + floyd_project + '/' + str(floyd_job) + '/output'
    try:
        with cd(output_dir):
            out = subprocess.run(clone_cmd, stdout=subprocess.PIPE, shell=True, check=True, stderr=subprocess.STDOUT).stdout.strip()
            if out.find(b'ERROR') != -1:
                raise subprocess.CalledProcessError(0, clone_cmd, out)
    except subprocess.CalledProcessError as e:
        print('FAILED TO CLONE FLOYD JOB OUTPUT:\n> ' + e.cmd + '\nSTDOUT:\n' + e.output.decode(CMD_ENCODING))
        shutil.rmtree(output_dir)
        return None

    # TODO: Modify tensorflow checkpoint file

    # Save best test score and hyperparameters to scores file
    print('Registering job to scores file...')
    entry = {'num': len(scores), 'floyd_job': floyd_job, 'model_name': os.path.basename(output_dir),
             'path': output_dir, 'floyd_url': os.path.join(floyd_project, str(floyd_job))}
    if hyperparameters is not None:
        entry['hyperparameters'] = hyperparameters
    if score is not None:
        entry['score'] = score
    with open(scores_path, 'wb') as file:
        np.save(file, np.append(scores, entry))

    # Delete Floyd job if asked so
    if delete_job:
        floyd_delete(floyd_project, floyd_job)
    return entry
