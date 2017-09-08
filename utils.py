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

# TODO: ignore robots.txt while downloading with wget in order to have all files, including files begining with a dot?
# TODO: snapshot ensembling proposed by https://arxiv.org/pdf/1704.00109.pdf

__all__ = ['tf_config', 'xavier_init', 'cosine_annealing', 'warm_restart', 'add_summary_values',
           'floyd_run', 'floyd_stop', 'floyd_delete', 'get_model_from_floyd', 'append_kaggle_score']

PREDS_FILE = 'preds.csv'

CMD_ENCODING = 'latin-1'
SCORES_FILE = 'scores.npz'
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))


def tf_config(allow_growth=True, **kwargs):
    config = tf.ConfigProto(**kwargs)
    config.gpu_options.allow_growth = allow_growth
    return config


def xavier_init(fan_in, fan_out, activation='relu'):
    if activation == 'relu':
        scale = math.sqrt(2.)
    elif activation == 'tanh':
        scale = 4.
    elif activation == 'sigmoid':
        scale = 1.
    else:
        raise ValueError('Invalid activation function for xavier initialization: "' + activation + '"')
    return tf.truncated_normal([fan_in, fan_out], stddev=scale * math.sqrt(2. / (fan_in + fan_out)))


def cosine_annealing(x, max_lr, min_lr):
    return (np.cos(np.pi * x) + 1.) / 2.


def warm_restart(epoch, t_0, max_lr, min_lr=1e-8, t_mult=2, annealing_fn=cosine_annealing):
    """ Stochastic gradient descent with warm restarts of learning rate (see https://arxiv.org/pdf/1608.03983.pdf) """
    def _cycle_length(c): return t_0 * t_mult ** c
    cycle = int(np.floor(np.log(- epoch / t_0 * (1 - t_mult) + 1) / np.log(t_mult)))
    cycle_begining = np.sum([_cycle_length(c) for c in range(0, cycle)]) if cycle > 0 else 0.
    x = (epoch - cycle_begining) / _cycle_length(cycle)
    lr = min_lr + (max_lr - min_lr) * annealing_fn(x, max_lr, min_lr)
    return lr, x == 0.


def add_summary_values(summary_writer, global_step=None, **values):
    if len(values) > 0:
        summary = tf.Summary()
        for name, value in values.items():
            summary.value.add(tag=name, simple_value=value)
        summary_writer.add_summary(summary, global_step=global_step)


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

    # Get output url from floyd job
    print('get job output url...')
    output_cmd = 'floyd output ' + os.path.join(floyd_project, str(floyd_job)) + ' -u'
    try:
        p = subprocess.run(output_cmd, stdout=subprocess.PIPE, shell=True, check=True, stderr=subprocess.STDOUT)
        url = p.stdout.strip().decode(CMD_ENCODING) + '/'
    except subprocess.CalledProcessError as e:
        print('FAILED TO GET FLOYD JOB OUTPUT URL:\n> ' + e.cmd + '\nSTDOUT:\n' + e.output.decode(CMD_ENCODING))
        return None

    # Download output directory
    print('downloading job output...')
    output_dir = os.path.join(models_dir, 'job_' + str(floyd_job))
    os.makedirs(output_dir)
    wget_cmd = 'wget -r -np -nH --cut-dirs=4 -P ' + output_dir + ' ' + url
    try:
        out = subprocess.run(wget_cmd, stdout=subprocess.PIPE, shell=True, check=True, stderr=subprocess.STDOUT).stdout.strip()
        if out.find(b'FINISHED') == -1:
            raise subprocess.CalledProcessError(0, wget_cmd, out)
    except subprocess.CalledProcessError as e:
        print('FAILED TO DOWNLOAD FLOYD JOB OUTPUT:\n> ' + e.cmd + '\nSTDOUT:\n' + e.output.decode(CMD_ENCODING))
        shutil.rmtree(output_dir)
        return None
    print('Cleaning output directory...')
    for root, dirnames, filenames in os.walk(output_dir):
        for filename in filenames:
            if os.path.basename(filename) == 'index.html' or os.path.basename(filename) == 'robots.txt':
                # Remove 'robot.txt' and 'index.html' files from output
                os.remove(os.path.join(root, filename))
            elif os.path.basename(filename) == PREDS_FILE:
                # Rename preds.csv
                os.rename(os.path.join(root, filename), os.path.join(root, os.path.dirname(filename), 'preds_' + str(floyd_job) + '.csv'))

    # Modify tensorflow checkpoint file
    # TODO: ...

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
        floyd_delete(floyd_job)
    return entry


def append_kaggle_score(kaggle_score, floyd_job=None, model_name=None):
    assert (floyd_job is None) != (model_name is None), 'You must either specify model_name xor floyd_job.'
    # Load score file
    scores_path = os.path.join(MODELS_DIR, SCORES_FILE)
    if not os.path.isfile(scores_path):
        print("Can't find score file.")
        return None
    scores = np.load(scores_path)
    # Search for concerned score entry
    field, value = ('floyd_job', floyd_job) if floyd_job is not None else ('model_name', model_name)
    found_scores = [(idx, s) for idx, s in enumerate(scores) if s[field] == value]
    if len(found_scores) == 0:
        print("Can't find specified model in score file, did you called 'get_model_from_floyd' first?")
        return None
    idx, score = found_scores[0]
    # Add kaggle score and save scores file
    score['kaggle_score'] = kaggle_score
    scores[idx] = score
    with open(scores_path, 'wb') as file:
        np.save(file, scores)
    return score


def delete_model(floyd_job=None, model_name=None):
    pass  # TODO: implement it
