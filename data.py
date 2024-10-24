from __future__ import division, print_function, absolute_import
import os, sys
import torch
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath('../'))


def get_feeddict(x, a, y, idx, name_prefix):
    x_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
        '%s_X_PLACEHOLDER:0' % name_prefix)
    a_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
        '%s_A_PLACEHOLDER:0' % name_prefix)
    y_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
        '%s_Y_PLACEHOLDER:0' % name_prefix)
    index_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
        '%s_INDEX_PLACEHOLDER:0' % name_prefix)
    feed_dict = {
        x_placeholder: x.numpy(),
        a_placeholder: a.numpy(),
        y_placeholder: y.numpy(),
        index_placeholder: idx.numpy()}
    return feed_dict


def get_placeholders(batch_size, num_frames, num_nodes, num_channels, 
                     multi_frame, multi_group,
                     name_prefix):
    x_placeholder = tf.compat.v1.placeholder(
        tf.float32,
        (batch_size, num_frames, num_nodes, num_channels + 4),
        name='%s_X_PLACEHOLDER' % name_prefix)
    a_placeholder = tf.compat.v1.placeholder(
        tf.float32,
        (batch_size, num_frames, num_nodes, num_nodes),
        name='%s_A_PLACEHOLDER' % name_prefix)
    y_placeholder = tf.compat.v1.placeholder(
        tf.int64,
        (batch_size),
        name='%s_Y_PLACEHOLDER' % name_prefix)
    index_placeholder = tf.compat.v1.placeholder(
        tf.int64,
        (batch_size),
        name='%s_INDEX_PLACEHOLDER' % name_prefix)
    if not multi_frame:
        if num_frames == 1:
            x_placeholder = tf.squeeze(x_placeholder, axis=1)
            a_placeholder = tf.squeeze(a_placeholder, axis=1)
        else:
            x_placeholder = tf.reshape(
                x_placeholder, 
                [-1, num_nodes, num_channels + 4])
            a_placeholder = tf.reshape(
                a_placeholder, 
                [-1, num_nodes, num_nodes])            
    else:
        if multi_group is not None:
            x_placeholder = tf.reshape(
                x_placeholder, 
                [batch_size*multi_group, num_frames // multi_group, num_nodes, num_channels + 4])
            a_placeholder = tf.reshape(
                a_placeholder, 
                [batch_size*multi_group, num_frames // multi_group, num_nodes, num_nodes])
    inputs = {
        'x': x_placeholder,
        'a': a_placeholder,
        'y': y_placeholder,
        'index': index_placeholder
    }
    return inputs

