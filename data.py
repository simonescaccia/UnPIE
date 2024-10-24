from __future__ import division, print_function, absolute_import
import os, sys
import torch
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath('../'))


def get_feeddict(x, a, y, name_prefix):
    x_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
        '%s_IMAGE_PLACEHOLDER:0' % name_prefix)
    a_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
        '%s_BBOX_PLACEHOLDER:0' % name_prefix)
    y_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
        '%s_LABEL_PLACEHOLDER:0' % name_prefix)
    feed_dict = {
        x_placeholder: x.numpy(),
        a_placeholder: a.numpy(),
        y_placeholder: y.numpy()}
    return feed_dict


def get_placeholders(batch_size, num_channels, name_prefix):
    x_placeholder = tf.compat.v1.placeholder(
        tf.float32,
        (batch_size, None, num_channels + 4),
        name='%s_IMAGE_PLACEHOLDER' % name_prefix)
    a_placeholder = tf.compat.v1.placeholder(
        tf.float32,
        (batch_size, None, None),
        name='%s_BBOX_PLACEHOLDER' % name_prefix)
    y_placeholder = tf.compat.v1.placeholder(
        tf.int64,
        (batch_size),
        name='%s_LABEL_PLACEHOLDER' % name_prefix)

    inputs = {
        'x': x_placeholder,
        'a': a_placeholder,
        'y': y_placeholder
    }
    return inputs
