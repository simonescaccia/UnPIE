import json
import os
import argparse
from pathlib import Path
import time
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

    def __init__(self, feature_extractor, data_path):
        self.data_path = data_path
        self.test_videos_path = 'PSI2.0_Test/videos'
        self.train_val_videos_path = 'PSI2.0_videos_Train-Val/videos'
        self.json_split_file_path = 'PSI2.0_Test/splits/PSI2_split.json'
        self.dataset_cache_path = data_path + '/dataset_cache'
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

    '''
    Database organization

    db = {
        'video_name': {
            'pedestrian_id': # track_id-#
            { 
                'frames': [0, 1, 2, ...], # target pedestrian appeared frames
                'cv_annotations': {
                    'track_id': track_id, 
                    'bbox': [xtl, ytl, xbr, ybr], 
                },
                'nlp_annotations': {
                    vid_uid_pair: {'intent': [], 'description': [], 'key_frame': []},
                    ...
                }
            }
        }
    }
    '''
    def create_database(self):
        for split_name in ['train', 'val', 'test']:
            with open(self.json_split_file_path, 'r') as f:
                datasplits = json.load(f)
            print(f"Initialize {split_name} database, {time.strftime("%d%b%Y-%Hh%Mm%Ss")}")
            # 1. Init db
            db = self.init_db(sorted(datasplits[split_name]))
            # 2. get intent, remove missing frames
            self.update_db_annotations(db)
            # 3. cut sequences, remove early frames before the first key frame, and after last key frame
            # cut_sequence(db, db_log, args)

            database_name = 'intent_database_' + split_name + '.pkl'
            if not os.path.exists(self.dataset_cache_path):
                os.makedirs(self.dataset_cache_path)
            with open(os.path.join(args.database_path, database_name), 'wb') as fid:
                pickle.dump(db, fid)

        print("Finished collecting database!")

    def init_db(self, video_list, db_log, args):
        db = {}
        # data_split = 'train' # 'train', 'val', 'test'
        dataroot = args.dataset_root_path
        # key_frame_folder = 'cognitive_annotation_key_frame'
        if args.dataset == 'PSI2.0':
            extended_folder = 'PSI2.0_TrainVal/annotations/cognitive_annotation_extended'
        elif args.dataset == 'PSI1.0':
            extended_folder = 'PSI1.0/annotations/cognitive_annotation_extended'

        for video_name in sorted(video_list):
            try:
                with open(os.path.join(dataroot, extended_folder, video_name, 'pedestrian_intent.json'), 'r') as f:
                    annotation = json.load(f)
            except:
                with open(db_log, 'a') as f:
                    f.write(f"Error loading {video_name} pedestrian intent annotation json \n")
                continue
            db[video_name] = {}
            for ped in annotation['pedestrians'].keys():
                cog_annotation = annotation['pedestrians'][ped]['cognitive_annotations']
                nlp_vid_uid_pairs = cog_annotation.keys()
                self.add_ped_case(db, video_name, ped, nlp_vid_uid_pairs)
        return db

    def add_ped_case(self, db, video_name, ped_name, nlp_vid_uid_pairs):
        if video_name not in db:
            db[video_name] = {}

        db[video_name][ped_name] = {  # ped_name is 'track_id' in cv-annotation
            'frames': None,  # [] list of frame_idx of the target pedestrian appear
            'cv_annotations': {
                'track_id': ped_name,
                'bbox': []  # [] list of bboxes, each bbox is [xtl, ytl, xbr, ybr]
            },
            'nlp_annotations': {
                # [vid_uid_pair: {'intent': [], 'description': [], 'key_frame': []}]
            }
        }
        for vid_uid in nlp_vid_uid_pairs:
            db[video_name][ped_name]['nlp_annotations'][vid_uid] = {
                'intent': [],
                'description': [],
                'key_frame': []
                # 0: not key frame (expanded from key frames with NLP annotations)
                # 1: key frame (labeled by NLP annotations)
            }

    def split_frame_lists(self, frame_list, bbox_list, threshold=60):
        # For a sequence of an observed pedestrian, split into slices based on missingframes
        frame_res = []
        bbox_res = []
        inds_res = []

        inds_split = [0]
        frame_split = [frame_list[0]]  # frame list
        bbox_split = [bbox_list[0]]  # bbox list
        for i in range(1, len(frame_list)):
            if frame_list[i] - frame_list[i - 1] == 1: # consistent
                inds_split.append(i)
                frame_split.append(frame_list[i])
                bbox_split.append(bbox_list[i])
            else:  # # next position frame is missing observed
                if len(frame_split) > threshold:  # only take the slices longer than threshold=max_track_length=60
                    inds_res.append(inds_split)
                    frame_res.append(frame_split)
                    bbox_res.append(bbox_split)
                    inds_split = []
                    frame_split = []
                    bbox_split = []
                else:  # ignore splits that are too short
                    inds_split = []
                    frame_split = []
                    bbox_split = []
        # break loop when i reaches the end of list
        if len(frame_split) > threshold:  # reach the end
            inds_res.append(inds_split)
            frame_res.append(frame_split)
            bbox_res.append(bbox_split)

        return frame_res, bbox_res, inds_res

    def get_intent_des(self, db, vname, pid, split_inds, cog_annt):
        # split_inds: the list of indices of the intent_annotations for the current split of pid in vname
        for vid_uid in cog_annt.keys():
            intent_list = cog_annt[vid_uid]['intent']
            description_list = cog_annt[vid_uid]['description']
            key_frame_list = cog_annt[vid_uid]['key_frame']

            nlp_vid_uid = vid_uid
            db[vname][pid]['nlp_annotations'][nlp_vid_uid]['intent'] = [intent_list[i] for i in split_inds]
            db[vname][pid]['nlp_annotations'][nlp_vid_uid]['description'] = [description_list[i] for i in split_inds]
            db[vname][pid]['nlp_annotations'][nlp_vid_uid]['key_frame'] = [key_frame_list[i] for i in split_inds]

    def update_db_annotations(self, db, db_log, args):
        dataroot = args.dataset_root_path
        # key_frame_folder = 'cognitive_annotation_key_frame'
        if args.dataset == 'PSI2.0':
            extended_folder = 'PSI2.0_TrainVal/annotations/cognitive_annotation_extended'
        elif args.dataset == 'PSI1.0':
            extended_folder = 'PSI1.0/annotations/cognitive_annotation_extended'

        video_list = sorted(db.keys())
        for video_name in video_list:
            ped_list = list(db[video_name].keys())
            tracks = list(db[video_name].keys())
            try:
                with open(os.path.join(dataroot, extended_folder, video_name, 'pedestrian_intent.json'), 'r') as f:
                    annotation = json.load(f)
            except:
                with open(db_log, 'a') as f:
                    f.write(f"Error loading {video_name} pedestrian intent annotation json \n")
                continue

            for pedId in ped_list:
                observed_frames = annotation['pedestrians'][pedId]['observed_frames']
                observed_bboxes = annotation['pedestrians'][pedId]['cv_annotations']['bboxes']
                cog_annotation = annotation['pedestrians'][pedId]['cognitive_annotations']
                if len(observed_frames) == observed_frames[-1] - observed_frames[0] + 1: # no missing frames
                    threshold = args.max_track_size # 16 for intent/driving decision; 60 for trajectory
                    if len(observed_frames) > threshold:
                        cv_frame_list = observed_frames
                        cv_frame_box = observed_bboxes
                        db[video_name][pedId]['frames'] = cv_frame_list
                        db[video_name][pedId]['cv_annotations']['bbox'] = cv_frame_box
                        get_intent_des(db, video_name, pedId, [*range(len(observed_frames))], cog_annotation)
                    else: # too few frames observed
                        # print("Single ped occurs too short.", video_name, pedId, len(observed_frames))
                        with open(db_log, 'a') as f:
                            f.write(f"Single ped occurs too short. {video_name}, {pedId}, {len(observed_frames)} \n")
                        del db[video_name][pedId]
                else: # missing frames exist
                    with open(db_log, 'a') as f:
                        f.write(f"missing frames bbox noticed! , {video_name}, {pedId}, {len(observed_frames)}, frames observed from , {observed_frames[-1] - observed_frames[0] + 1} \n")
                    threshold = args.max_track_size  # 60
                    cv_frame_list, cv_frame_box, cv_split_inds = split_frame_lists(observed_frames, observed_bboxes, threshold)
                    if len(cv_split_inds) == 0:
                        with open(db_log, 'a') as f:
                            f.write(f"{video_name}, {pedId}, After removing missing frames, no split left! \n")

                        del db[video_name][pedId]
                    elif len(cv_split_inds) == 1:
                        db[video_name][pedId]['frames'] = cv_frame_list[0]
                        db[video_name][pedId]['cv_annotations']['bbox'] = cv_frame_box[0]
                        get_intent_des(db, video_name, pedId, cv_split_inds[0], cog_annotation)
                    else:
                        # multiple splits left after removing missing box frames
                        with open(db_log, 'a') as f:
                            f.write(f"{len(cv_frame_list)} splits: , {[len(s) for s in cv_frame_list]} \n")
                        nlp_vid_uid_pairs = db[video_name][pedId]['nlp_annotations'].keys()
                        for i in range(len(cv_frame_list)):
                            ped_splitId = pedId + '-' + str(i)
                            add_ped_case(db, video_name, ped_splitId, nlp_vid_uid_pairs)
                            db[video_name][ped_splitId]['frames'] = cv_frame_list[i]
                            db[video_name][ped_splitId]['cv_annotations']['bbox'] = cv_frame_box[i]
                            get_intent_des(db, video_name, ped_splitId, cv_split_inds[i], cog_annotation)
                            if len(db[video_name][ped_splitId]['nlp_annotations'][list(db[video_name][ped_splitId]['nlp_annotations'].keys())[0]]['intent']) == 0:
                                raise Exception("ERROR!")
                        del db[video_name][pedId] # no pedestrian list left, remove this video
                tracks.remove(pedId)
            if len(db[video_name].keys()) < 1: # has no valid ped sequence! Remove this video!")
                with open(db_log, 'a') as f:
                    f.write(f"!!!!! Video, {video_name}, has no valid ped sequence! Remove this video! \n")
                del db[video_name]
            if len(tracks) > 0:
                with open(db_log, 'a') as f:
                    f.write(f"{video_name} missing pedestrians annotations: {tracks}  \n")
