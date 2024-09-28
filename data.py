from __future__ import division, print_function, absolute_import
import os, sys
import torch
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath('../'))


def get_feeddict(image, label, index, name_prefix='TRAIN'):
    image_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
            '%s_IMAGE_PLACEHOLDER:0' % name_prefix)
    label_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
            '%s_LABEL_PLACEHOLDER:0' % name_prefix)
    index_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
            '%s_INDEX_PLACEHOLDER:0' % name_prefix)
    feed_dict = {
            image_placeholder: image.numpy(),
            label_placeholder: label.numpy(),
            index_placeholder: index.numpy()}
    return feed_dict


def get_placeholders(
        batch_size, num_frames, 
        emb_dim,
        name_prefix, multi_frame, multi_group):
    image_placeholder = tf.compat.v1.placeholder(
            tf.uint8,
            (batch_size, num_frames, emb_dim),
            name='%s_IMAGE_PLACEHOLDER' % name_prefix)
    label_placeholder = tf.compat.v1.placeholder(
            tf.int64,
            (batch_size),
            name='%s_LABEL_PLACEHOLDER' % name_prefix)
    index_placeholder = tf.compat.v1.placeholder(
            tf.int64,
            (batch_size),
            name='%s_INDEX_PLACEHOLDER' % name_prefix)
    if not multi_frame:
        if num_frames == 1:
            image_placeholder = tf.squeeze(image_placeholder, axis=1)
        else:
            image_placeholder = tf.reshape(
                    image_placeholder, 
                    [-1, emb_dim])
    else:
        if multi_group is not None:
            image_placeholder = tf.reshape(
                    image_placeholder, 
                    [batch_size*multi_group, num_frames // multi_group, \
                            emb_dim])
    inputs = {
            'image': image_placeholder,
            'label': label_placeholder,
            'index': index_placeholder}
    return inputs
