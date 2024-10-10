from __future__ import division, print_function, absolute_import
import os, sys
import torch
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath('../'))


def get_feeddict(image, bbox, label, index, name_prefix):
    image_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
            '%s_IMAGE_PLACEHOLDER:0' % name_prefix)
    bbox_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
            '%s_BBOX_PLACEHOLDER:0' % name_prefix)
    label_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
            '%s_LABEL_PLACEHOLDER:0' % name_prefix)
    index_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
            '%s_INDEX_PLACEHOLDER:0' % name_prefix)
    feed_dict = {
            image_placeholder: image.numpy(),
            bbox_placeholder: bbox.numpy(),
            label_placeholder: label.numpy(),
            index_placeholder: index.numpy()}
    return feed_dict


def get_placeholders(
        batch_size, num_frames, 
        crop_size, num_channels,
        name_prefix, multi_frame, multi_group):
    image_placeholder = tf.compat.v1.placeholder(
            tf.float32,
            (batch_size, num_frames, crop_size, crop_size, num_channels),
            name='%s_IMAGE_PLACEHOLDER' % name_prefix)
    bbox_placeholder = tf.compat.v1.placeholder(
            tf.float32,
            (batch_size, num_frames, 4),
            name='%s_BBOX_PLACEHOLDER' % name_prefix)
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
                    [-1, crop_size, crop_size, num_channels])
    else:
        if multi_group is not None:
            image_placeholder = tf.reshape(
                    image_placeholder, 
                    [batch_size*multi_group, num_frames // multi_group, \
                            crop_size, crop_size, num_channels])
    inputs = {
            'image': image_placeholder,
            'bbox': bbox_placeholder,
            'label': label_placeholder,
            'index': index_placeholder}
    return inputs
