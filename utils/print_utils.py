import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pandas as pd
import tensorflow as tf
from utils.pie_utils import jitter_bbox, squarify

num_dashes = 100

def write_dict(d, filename):
    with open(filename, 'w') as f:
        _write_dict(d, f)
        
def _write_dict(d, f, indent=0):
    for k, v in d.items():
        if isinstance(v, dict):
            f.write('  '*indent + f"{k}:\n")
            _write_dict(v, f, indent+1)
        else:
            f.write('  '*indent + f"{k}: {v}\n")

def print_separator(message=None, space=True, top_new_line=True, bottom_new_line=True):
    if space and top_new_line:
        print()
    print('-'*num_dashes)
    if message:
        print(message)
    if space and bottom_new_line:
        print()
    return

def print_model_size(model, model_name):
    # Calculate the number of parameters
    num_params = model.count_params()
    # Assuming each parameter is 4 bytes (32 bits)
    memory_bytes = num_params * 4
    # Convert to megabytes
    memory_mb = memory_bytes / (1024 ** 2)
    print('')
    print(model_name, f" memory: {memory_mb:.2f} MB\n")

def _plot_image_with_bbox(image, df):
    # Convert CV image to PIL image
    image1 = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image1)
    
    for i, row in df.iterrows():
        image_i = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        b = list(row['bbox'])
        bbox = jitter_bbox([b],'enlarge', 2, image=image_pil)[0]
        bbox = squarify(bbox, 1, image_pil.size[0])
        bbox = list(map(int,bbox[0:4]))
        cv2.rectangle(image_i, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        plt.imshow(image_i)
        plt.axis('off')
        plt.show()

    plt.imshow(image2)
    plt.axis('off')
    plt.show()

def plot_image(image, df: pd.DataFrame, ped_type, traffic_type, type_column, set_id, vid, frame_num):
    # good frames:
    # sid, vid, frameid
    # 6, 2, 14053, wait cross
    # 6, 4, 10147, wait cross bike
    # 6, 1, 14272, social
    # 6, 1, 16748
    save = True
    if save:
        # save frame
        target_frame = 16748
        target_video = 'video_0001'
        target_set = 'set06'
        if set_id == target_set and vid == target_video and (frame_num == target_frame-1 or frame_num == target_frame or frame_num == target_frame+1):
            os.system('mkdir -p images')
            file_path = f'images/{set_id}-{vid}-{frame_num}.png'
            _plot_image(image, save, file_path)
    else:
        # show frames
        len_min_traffic = 0
        len_min_ped = 1
        len_max_ped = 1
        if set_id != 'set06':
            return
        values = df[type_column].value_counts()
        values_keys = values.keys()
        if ped_type in values_keys and traffic_type in values_keys and \
            values[traffic_type] > len_min_traffic and \
            values[ped_type] >= len_min_ped and values[ped_type] <= len_max_ped:
            print("frame: ", frame_num)
            _plot_image(image, save)

def _plot_image(image, save, file_path=None):
    # Convert CV image to PIL image
    image2 = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.imshow(image2)
    plt.axis('off')
    if save:
        print("Saving ", file_path)
        cv2.imwrite(file_path, image)
    else:    
        plt.show()

def print_gpu_memory_2():
    # get tensorflow memory info
    info = tf.config.experimental.get_memory_info('GPU:0')
    # print memory info in Bytes
    print(f"GPU memory current: {info['current']} Bytes, peak: {info['peak']} Bytes")

import subprocess as sp
import os

def print_gpu_memory(flag):
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    print(flag, ": GPU memory free: ", memory_free_values)
