from __future__ import division, print_function, absolute_import
import os, sys
import torch
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath('../'))


def get_feeddict(image, bbox, objs_img, objs_bbox, other_peds_img, other_peds_bbox, label, index, name_prefix):
    image_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
        '%s_IMAGE_PLACEHOLDER:0' % name_prefix)
    bbox_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
        '%s_BBOX_PLACEHOLDER:0' % name_prefix)
    objs_img_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
        '%s_OBJS_IMG_PLACEHOLDER:0' % name_prefix)
    objs_bbox_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
        '%s_OBJS_BBOX_PLACEHOLDER:0' % name_prefix)
    other_peds_img_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
        '%s_OTHER_PEDS_IMG_PLACEHOLDER:0' % name_prefix)
    other_peds_bbox_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
        '%s_OTHER_PEDS_BBOX_PLACEHOLDER:0' % name_prefix)
    label_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
        '%s_LABEL_PLACEHOLDER:0' % name_prefix)
    index_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
        '%s_INDEX_PLACEHOLDER:0' % name_prefix)
    feed_dict = {
        image_placeholder: image.numpy(),
        bbox_placeholder: bbox.numpy(),
        objs_img_placeholder: objs_img.numpy(),
        objs_bbox_placeholder: objs_bbox.numpy(),
        other_peds_img_placeholder: other_peds_img.numpy(),
        other_peds_bbox_placeholder: other_peds_bbox.numpy(),
        label_placeholder: label.numpy(),
        index_placeholder: index.numpy()}
    return feed_dict


def get_placeholders(
        batch_size, num_frames, 
        crop_size, num_channels,
        name_prefix, multi_frame, multi_group):
    image_placeholder = tf.compat.v1.placeholder(
        tf.float32,
        (batch_size, num_frames, num_channels),
        name='%s_IMAGE_PLACEHOLDER' % name_prefix)
    bbox_placeholder = tf.compat.v1.placeholder(
        tf.float32,
        (batch_size, num_frames, 4),
        name='%s_BBOX_PLACEHOLDER' % name_prefix)
    objs_img_placeholder = tf.compat.v1.placeholder(
        tf.float32,
        (batch_size, num_frames, None, num_channels),
        name='%s_OBJS_IMG_PLACEHOLDER' % name_prefix)
    objs_bbox_placeholder = tf.compat.v1.placeholder(
        tf.float32,
        (batch_size, num_frames, None ,4),
        name='%s_OBJS_BBOX_PLACEHOLDER' % name_prefix)
    other_peds_img_placeholder = tf.compat.v1.placeholder(
        tf.float32,
        (batch_size, num_frames, None, num_channels),
        name='%s_OTHER_PEDS_IMG_PLACEHOLDER' % name_prefix)
    other_peds_bbox_placeholder = tf.compat.v1.placeholder(
        tf.float
        (batch_size, num_frames, None, 4),
        name='%s_OTHER_PEDS_BBOX_PLACEHOLDER' % name_prefix)
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
            bbox_placeholder = tf.squeeze(bbox_placeholder, axis=1)
            objs_img_placeholder = tf.squeeze(objs_img_placeholder, axis=1)
            objs_bbox_placeholder = tf.squeeze(objs_bbox_placeholder, axis=1)
            other_peds_img_placeholder = tf.squeeze(other_peds_img_placeholder, axis=1)
            other_peds_bbox_placeholder = tf.squeeze(other_peds_bbox_placeholder, axis=1)
        else:
            image_placeholder = tf.reshape(
                image_placeholder, 
                [-1, num_channels])
            bbox_placeholder = tf.reshape(
                bbox_placeholder, 
                [-1, 4])
            objs_img_placeholder = tf.reshape(
                objs_img_placeholder,
                [-1, None, num_channels])
            objs_bbox_placeholder = tf.reshape(
                objs_bbox_placeholder,
                [-1, None, 4])
            other_peds_img_placeholder = tf.reshape(
                other_peds_img_placeholder,
                [-1, None, num_channels])
            other_peds_bbox_placeholder = tf.reshape(
                other_peds_bbox_placeholder,
                [-1, None, 4])
    else:
        if multi_group is not None:
            image_placeholder = tf.reshape(
                image_placeholder, 
                [batch_size*multi_group, num_frames // multi_group, num_channels])
            bbox_placeholder = tf.reshape(
                bbox_placeholder, 
                [batch_size*multi_group, num_frames // multi_group, 4])
            objs_img_placeholder = tf.reshape(
                objs_img_placeholder,
                [batch_size*multi_group, num_frames // multi_group, None, num_channels])
            objs_bbox_placeholder = tf.reshape(
                objs_bbox_placeholder,
                [batch_size*multi_group, num_frames // multi_group, None, 4])
            other_peds_img_placeholder = tf.reshape(
                other_peds_img_placeholder,
                [batch_size*multi_group, num_frames // multi_group, None, num_channels])
            other_peds_bbox_placeholder = tf.reshape(
                other_peds_bbox_placeholder,
                [batch_size*multi_group, num_frames // multi_group, None, 4])

    inputs = {
            'image': image_placeholder,
            'bbox': bbox_placeholder,
            'label': label_placeholder,
            'index': index_placeholder,
            'objs_img': objs_img_placeholder,
            'objs_bbox': objs_bbox_placeholder,
            'other_peds_img': other_peds_img_placeholder,
            'other_peds_bbox': other_peds_bbox_placeholder}
    return inputs
