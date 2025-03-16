import json
import os
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm
import sys

from dataset.pretrained_extractor import PretrainedExtractor

TRAIN = 'train'
VAL = 'val'
TEST = 'test'
ALL = 'all'
TRAIN_VAL = 'train_val'

class PSI(object):

    def __init__(self, feature_extractor):
        self.test_videos_path = 'PSI2.0_Test/videos'
        self.train_val_videos_path = 'PSI2.0_videos_Train-Val/videos'
        self.json_split_file_path = 'PSI2.0_Test/splits/PSI2_split.json'
        self.video_set_nums = {
            TRAIN: ['set01'],
            VAL: ['set02'],
            TEST: ['set03'],
            ALL: ['set01', 'set02', 'set03']
        }

        self.feature_extractor = feature_extractor

    def load_split_json(self):
        with open(self.json_split_file_path, 'r') as f:
            split_json = json.load(f)
        return split_json

    def get_video_split(self, split_name, video_name):
        if split_name != TRAIN_VAL:
            return split_name
        else:
            for split, videos in self.split_json.items():
                if video_name in videos:
                    return split

    '''Given video path, extract frames for all videos. Check if frames exist first.'''
    def extract_images_and_save_features(self, split):
        self.pretrained_extractor = PretrainedExtractor(self.feature_extractor) # Create extractor model
        self.split_json = self.load_split_json()

        if split == ALL:
            video_paths = [self.train_val_videos_path, self.test_videos_path]
            splits = [TRAIN_VAL, TEST]
        elif split == TRAIN_VAL:
            video_paths = [self.train_val_videos_path]
            splits = [TRAIN_VAL]
        elif split == TEST:
            video_paths = [self.test_videos_path]
            splits = [TEST]
        else:
            raise ValueError('Invalid split name')
          
        for video_path, split_name in tqdm(zip(video_paths, splits)):
            for video in tqdm(sorted(os.listdir(video_path))):
                name = video.split('.mp4')[0]
                video_target = os.path.join(video_path, video)
                split_target = self.get_video_split(split_name, name)

                try:
                    vidcap = cv2.VideoCapture(video_target)
                    if not vidcap.isOpened():
                        raise Exception(f'Cannot open file {video}')
                except Exception as e:
                    raise e

                success, frame = vidcap.read()
                cur_frame = 0
                while(success):
                    frame_num = str(cur_frame).zfill(3)
                    cv2.imwrite(os.path.join(frames_target, f'{frame_num}.jpg'), frame)
                    success, frame = vidcap.read()
                    cur_frame += 1
                vidcap.release()
                # break
