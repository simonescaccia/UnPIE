"""
The code implementation of the paper:

A. Rasouli, I. Kotseruba, T. Kunic, and J. Tsotsos, "PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and
Trajectory Prediction", ICCV 2019.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Updated by: Simone Scaccia
"""
import os
import shutil
import sys
import cv2
import PIL
import pickle
from PIL import Image
from keras.utils import load_img
import numpy as np


def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)

    block = int(round(barLength*progress))
    text = "\r[{}] {:0.2f}% {}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def merge_directory(source_dir, dest_dir):
    for file in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file)
        dest_file = os.path.join(dest_dir, file)
        if not os.path.exists(dest_file):
            shutil.move(source_file, dest_file)
        elif os.path.isdir(source_file):
            merge_directory(source_file, dest_file)
        else:
            print('File already exists in the destination directory')
            print('File: %s' % dest_file)
    if len(os.listdir(source_dir)) == 0:
        os.rmdir(source_dir)
    else:
        print('Cannot merge directory %s to %s' % (source_file, dest_file))  


######### DATA UTILITIES ##############

def get_folder_from_set(set_id, video_set_nums):
    """
    Returns the folder name from the set id
    :param set_id: Set id
    :return: Folder name
    """
    for folder in video_set_nums:
        if set_id in video_set_nums[folder]:
            return folder
    print('\nSet id not found in the video set! Please check self.video_set_nums\n')

def get_path(dataset_path, model_name, data_subset, data_type, feature_type):
    """
    A path generator method for saving model and config data. Creates directories
    as needed.
    :param model_name: model name
    :param data_subset: all, train, test or val
    :param data_type: type of the data (e.g. features_context_pad_resize)
    :param feature_type: type of the feature (e.g. ped, traffic)
    :return: The full path for the save folder
    """
    root = os.path.join(dataset_path, 'data')
    save_path = os.path.join(root, model_name, data_type, feature_type, data_subset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path

def extract_and_save(b, ped_id, set_id, vid_id, img_name, image, feature_type, dataset_path, pretrained_extractor, data_opts, video_set_nums, feat_input_size):
    """
    @author: Simone Scaccia
    Extracts features from images and saves them on hard drive
    :param img_path: The path to the image
    :param b: The bounding box of the pedestrian
    :param ped_id: The pedestrian id
    :param set_id: The set id
    :param vid_id: The video id
    :param img_name: The image name
    :param image: The image
    :param save_path: The path to save the features
    """
    save_path = get_path(
        dataset_path=dataset_path,
        data_type='features'+'_'+data_opts['crop_type']+'_'+data_opts['crop_mode'], # images    
        model_name=pretrained_extractor.model_name,
        data_subset='all',
        feature_type=feature_type)
    dest_folder = get_folder_from_set(set_id, video_set_nums)
    dest_path = get_path(
        dataset_path=dataset_path,
        data_type='features'+'_'+data_opts['crop_type']+'_'+data_opts['crop_mode'], # images    
        model_name=pretrained_extractor.model_name,
        data_subset=dest_folder,
        feature_type=feature_type)
    img_save_folder = os.path.join(save_path, set_id, vid_id)
    img_save_path = os.path.join(img_save_folder, img_name+'_'+ped_id+'.pkl')
    img_dest_folder = os.path.join(dest_path, set_id, vid_id)
    img_dest_path = os.path.join(img_dest_folder, img_name+'_'+ped_id+'.pkl')
    if not os.path.exists(img_dest_path) and not os.path.exists(img_save_path):
        # Convert CV image to PIL image
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        bbox = jitter_bbox([b],'enlarge', 2, image=image)[0]
        bbox = squarify(bbox, 1, image.size[0])
        bbox = list(map(int,bbox[0:4]))
        cropped_image = image.crop(bbox)
        img_data = img_pad(cropped_image, mode='pad_resize', size=feat_input_size)                    
        expanded_img = pretrained_extractor.preprocess(img_data)
        img_features = pretrained_extractor(expanded_img)
        if not os.path.exists(img_save_folder):
            os.makedirs(img_save_folder)
        with open(img_save_path, 'wb') as fid:
            pickle.dump(img_features, fid, pickle.HIGHEST_PROTOCOL) 

def img_pad(img, mode = 'warp', size = 224):
    '''
    Pads a given image.
    Crops and/or pads a image given the boundries of the box needed
    img: the image to be coropped and/or padded
    bbox: the bounding box dimensions for cropping
    size: the desired size of output
    mode: the type of padding or resizing. The modes are,
        warp: crops the bounding box and resize to the output size
        same: only crops the image
        pad_same: maintains the original size of the cropped box  and pads with zeros
        pad_resize: crops the image and resize the cropped box in a way that the longer edge is equal to
        the desired output size in that direction while maintaining the aspect ratio. The rest of the image is
        padded with zeros
        pad_fit: maintains the original size of the cropped box unless the image is biger than the size in which case
        it scales the image down, and then pads it
    '''
    assert(mode in ['same', 'warp', 'pad_same', 'pad_resize', 'pad_fit']), 'Pad mode %s is invalid' % mode
    image = img.copy()
    if mode == 'warp':
        warped_image = image.resize((size,size),PIL.Image.NEAREST)
        return warped_image
    elif mode == 'same':
        return image
    elif mode in ['pad_same','pad_resize','pad_fit']:
        img_size = image.size  # size is in (width, height)
        ratio = float(size)/max(img_size)
        if mode == 'pad_resize' or  \
            (mode == 'pad_fit' and (img_size[0] > size or img_size[1] > size)):
            img_size = tuple([int(img_size[0]*ratio),int(img_size[1]*ratio)])
            image = image.resize(img_size, PIL.Image.NEAREST)
        padded_image = PIL.Image.new("RGB", (size, size))
        padded_image.paste(image, ((size-img_size [0])//2,
                    (size-img_size [1])//2))
        return padded_image

def bbox_center(bbox):
    '''
    Returns the center of the bounding box
    '''
    return [(bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2] # [width, height]

def squarify(bbox, ratio, img_width):
    """
    Changes the ratio of bounding boxes to a fixed ratio
    :param bbox: Bounding box
    :param ratio: Ratio to be changed to
    :param img_width: Image width
    :return: Squarified boduning box
    """
    width = abs(bbox[0] - bbox[2])
    height = abs(bbox[1] - bbox[3])
    width_change = height * ratio - width

    bbox[0] = bbox[0] - width_change / 2
    bbox[2] = bbox[2] + width_change / 2

    if bbox[0] < 0:
        bbox[0] = 0

    # check whether the new bounding box goes beyond image boarders
    # If this is the case, the bounding box is shifted back
    if bbox[2] > img_width:
        bbox[0] = bbox[0] - bbox[2] + img_width
        bbox[2] = img_width
    return bbox

def bbox_sanity_check(img, bbox):
    '''
    This is to confirm that the bounding boxes are within image boundaries.
    If this is not the case, modifications is applied.
    This is to deal with inconsistencies in the annotation tools
    '''
    img_width, img_heigth = img.size
    if bbox[0] < 0:
        bbox[0] = 0.0
    if bbox[1] < 0:
        bbox[1] = 0.0
    if bbox[2] >= img_width:
        bbox[2] = img_width - 1
    if bbox[3] >= img_heigth:
        bbox[3] = img_heigth - 1
    return bbox


def jitter_bbox(bbox, mode, ratio, image = None, img_path = None):
    '''
    This method jitters the position or dimentions of the bounding box.
    mode: 'same' returns the bounding box unchanged
          'enlarge' increases the size of bounding box based on the given ratio.
          'random_enlarge' increases the size of bounding box by randomly sampling a value in [0,ratio)
          'move' moves the center of the bounding box in each direction based on the given ratio
          'random_move' moves the center of the bounding box in each direction by randomly sampling a value in [-ratio,ratio)
    ratio: The ratio of change relative to the size of the bounding box. For modes 'enlarge' and 'random_enlarge'
           the absolute value is considered.
    Note: Tha ratio of change in pixels is calculated according to the smaller dimension of the bounding box.
    '''
    assert(mode in ['same','enlarge','move','random_enlarge','random_move']), \
            'mode %s is invalid.' % mode

    if mode == 'same':
        return bbox

    if img_path is not None:
        img = load_img(img_path)
    else:
        img = image

    img_width, img_heigth = img.size

    if mode in ['random_enlarge', 'enlarge']:
        jitter_ratio  = abs(ratio)
    else:
        jitter_ratio  = ratio

    if mode == 'random_enlarge':
        jitter_ratio = np.random.random_sample()*jitter_ratio
    elif mode == 'random_move':
        # for ratio between (-jitter_ratio, jitter_ratio)
        # for sampling the formula is [a,b), b > a,
        # random_sample * (b-a) + a
        jitter_ratio = np.random.random_sample() * jitter_ratio * 2 - jitter_ratio

    jit_boxes = []
    for b in bbox:
        bbox_width = b[2] - b[0]
        bbox_height = b[3] - b[1]

        width_change = bbox_width * jitter_ratio
        height_change = bbox_height * jitter_ratio

        if width_change < height_change:
            height_change = width_change
        else:
            width_change = height_change

        if mode in ['enlarge','random_enlarge']:
            b[0] = b[0] - width_change //2
            b[1] = b[1] - height_change //2
        else:
            b[0] = b[0] + width_change //2
            b[1] = b[1] + height_change //2

        b[2] = b[2] + width_change //2
        b[3] = b[3] + height_change //2

        # Checks to make sure the bbox is not exiting the image boundaries
        b =  bbox_sanity_check(img, b)
        jit_boxes.append(b)
    # elif crop_opts['mode'] == 'border_only':
    return jit_boxes

