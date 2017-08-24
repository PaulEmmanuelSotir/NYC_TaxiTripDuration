#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Utils

.. See https://github.com/PaulEmmanuelSotir/NYC_TaxiTripDuration
"""
import re
import os
import json
import math
import shutil
import subprocess
import numpy as np
import tensorflow as tf

__all__ = ['tf_config', 'visualize_weights']

def tf_config(allow_growth=True, **kwargs):
    config = tf.ConfigProto(**kwargs)
    config.gpu_options.allow_growth = allow_growth
    return config

def xavier_init(fan_in, fan_out):
    return tf.random_normal([fan_in, fan_out], stddev=math.sqrt(3. / (fan_in + fan_out)))
