import json
import os
import argparse
from pathlib import Path
import pickle
import time
import cv2
import numpy as np
from tqdm import tqdm
import sys

from dataset.pretrained_extractor import PretrainedExtractor
from utils.data_utils import get_ped_info_per_image

TRAIN = 'train'
VAL = 'val'
TEST = 'test'
ALL = 'all'
TRAIN_VAL = 'train_val'

class PSI(object):

    def __init__(self, feature_extractor, data_path, data_opts, object_class_list):
        self.data_opts = data_opts
        self.data_path = data_path
        self.test_path = 'PSI2.0_Test'
        self.train_val_path = 'PSI2.0_TrainVal'
        self.test_videos_path = self.test_path + '/videos'
        self.train_val_videos_path = 'PSI2.0_videos_Train-Val/videos'
        self.json_split_file_path = self.train_val_path + '/splits/PSI2_split.json'
        self.dataset_cache_path = data_path + '/dataset_cache'
        self.extended_folder = '/annotations/cognitive_annotation_extended'
        self.cv_annotation_folder = '/annotations/cv_annotation'
        self.video_set_nums = {
            TRAIN: ['set01'],
            VAL: ['set02'],
            TEST: ['set03'],
            ALL: ['set01', 'set02', 'set03']
        }
        self.object_class_list = object_class_list
        self.feature_extractor = feature_extractor
        self.intent_type = 'mean'

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

    def get_unique_values(self, data, seen=None):
        if seen is None:
            seen = set()
        
        if isinstance(data, list):
            for item in data:
                self.get_unique_values(item, seen)
        else:
            seen.add(data)
        
        return seen


    '''Given video path, extract frames for all videos. Check if frames exist first.'''
    def extract_images_and_save_features(self, split):
        self.pretrained_extractor = PretrainedExtractor(self.feature_extractor) # Create extractor model
        annotation_train, annotation_val, annotation_test = self.generate_database()
        test_seq = self.generate_data_sequence('trian', annotation_test)
        ped_objs_dataframe = get_ped_info_per_image(
            images=test_seq['frame'],
            bboxes=test_seq['bbox'],
            ped_ids=test_seq['ped_id'],
            obj_bboxes=test_seq['objs_bboxes'],
            obj_ids=test_seq['objs_ids'],
            other_ped_bboxes=[],
            other_ped_ids=[],
            ped_type=self.ped_type,
            traffic_type=self.traffic_type)

        print("Objects: ", self.get_unique_values(test_seq['objs_classes']))

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

    cv_db = {
        'video_name': *video_name*,
        'frames': {
            'frame_*frameId*': {
                'speed(km/hr)': str, # e.g., "72"
                'gps': [str, str], # [N/S, W/E] e.g., ["N39.602013","W86.159046"]
                'time': str, # 'hh:mm:ss', e.g., "16:57:20"
                'cv_annotation': {
                    '*objType*_track_*trackId*': {
                        'object_type': str, # e.g., 'car', 'pedestrian', etc.
                        'track_id': str, # e.g, 'track_3'
                        'bbox': [float, float, float, float], # [xtl, ytl, xbr, ybr]
                        'observed_frames': [int, int, int, ...], # e.g., [153, 154, 155, ...]
                    }
                }
            }
        }
    }
    '''
    def generate_database(self):
        database_name = "intent_database_{split_name}.pkl"
        annotation_splits = []
        for split in ['train', 'val', 'test']:
            database_path = os.path.join(self.dataset_cache_path, database_name.format(split_name=split))
            if os.path.exists(database_path):
                with open(database_path, 'rb') as f:
                    annotation_splits.append(pickle.load(f))
            else:
                annotation_splits.append(self.create_database(split))
        return annotation_splits[0], annotation_splits[1], annotation_splits[2]

    def create_database(self, split_name):
        with open(self.json_split_file_path, 'r') as f:
            datasplits = json.load(f)
        print(f"Initialize {split_name} database, {time.strftime('%d%b%Y-%Hh%Mm%Ss')}")

        split_extended_folder = self.test_path + self.extended_folder if split_name == 'test' else self.train_val_path + self.extended_folder
        split_cv_annotation_folder = self.test_path + self.cv_annotation_folder if split_name == 'test' else self.train_val_path + self.cv_annotation_folder

        # 1. Init db
        db = self.init_db(sorted(datasplits[split_name]), split_extended_folder)
        # 2. get intent, remove missing frames
        self.update_db_annotations(db, split_extended_folder, split_cv_annotation_folder)
        # 3. cut sequences, remove early frames before the first key frame, and after last key frame
        # cut_sequence(db, db_log, args)

        database_name = 'intent_database_' + split_name + '.pkl'
        if not os.path.exists(self.dataset_cache_path):
            os.makedirs(self.dataset_cache_path)
        with open(os.path.join(self.dataset_cache_path, database_name), 'wb') as fid:
            pickle.dump(db, fid)

        return db

    def init_db(self, video_list, split_extended_folder):
        db = {}

        for video_name in tqdm(sorted(video_list)):
            # Pedestrian intent annotation
            try:
                with open(os.path.join(self.data_path, split_extended_folder, video_name, 'pedestrian_intent.json'), 'r') as f:
                    ped_annotation = json.load(f)
            except:
                print(f"Error loading {video_name} pedestrian intent annotation json \n")
                continue
            
            db[video_name] = {}
            for ped in ped_annotation['pedestrians'].keys():
                cog_annotation = ped_annotation['pedestrians'][ped]['cognitive_annotations']
                nlp_vid_uid_pairs = cog_annotation.keys()
                self.add_ped_case(db, video_name, ped, nlp_vid_uid_pairs)
        return db

    def update_db_annotations(self, db, split_extended_folder, split_cv_annotation_folder):
        video_list = sorted(db.keys())
        for video_name in tqdm(video_list):
            ped_list = list(db[video_name].keys())
            tracks = list(db[video_name].keys())
            try:
                with open(os.path.join(self.data_path, split_extended_folder, video_name, 'pedestrian_intent.json'), 'r') as f:
                    annotation = json.load(f)
            except:
                print(f"Error loading {video_name} pedestrian intent annotation json \n")
                continue
            try:
                # Check cv annotation exists
                if os.path.exists(os.path.join(self.data_path, split_cv_annotation_folder, video_name, 'cv_annotation.json')):
                    with open(os.path.join(self.data_path, split_cv_annotation_folder, video_name, 'cv_annotation.json'), 'r') as f:
                        cv_annotation = json.load(f)
            except:
                print(f"Error loading {video_name} cv annotation json \n")

            # Pedestrian intent annotation
            for pedId in ped_list:
                observed_frames = annotation['pedestrians'][pedId]['observed_frames']
                observed_bboxes = annotation['pedestrians'][pedId]['cv_annotations']['bboxes']
                cog_annotation = annotation['pedestrians'][pedId]['cognitive_annotations']
                if len(observed_frames) == observed_frames[-1] - observed_frames[0] + 1: # no missing frames
                    threshold = self.data_opts['max_size_observe'] # 15 for intent decision
                    if len(observed_frames) > threshold:
                        cv_frame_list = observed_frames
                        cv_frame_box = observed_bboxes
                        db[video_name][pedId]['frames'] = cv_frame_list
                        db[video_name][pedId]['cv_annotations']['bbox'] = cv_frame_box
                        self.get_intent_des(db, video_name, pedId, [*range(len(observed_frames))], cog_annotation)
                    else: # too few frames observed
                        # print("Single ped occurs too short.", video_name, pedId, len(observed_frames))
                        # with open(db_log, 'a') as f:
                        #     f.write(f"Single ped occurs too short. {video_name}, {pedId}, {len(observed_frames)} \n")
                        del db[video_name][pedId]
                else: # missing frames exist
                    # print(f"missing frames bbox noticed! , {video_name}, {pedId}, {len(observed_frames)}, frames observed from , {observed_frames[-1] - observed_frames[0] + 1} \n")
                    threshold = self.data_opts['max_size_observe']
                    cv_frame_list, cv_frame_box, cv_split_inds = self.split_frame_lists(observed_frames, observed_bboxes, threshold)
                    if len(cv_split_inds) == 0:
                        # print(f"{video_name}, {pedId}, After removing missing frames, no split left! \n")

                        del db[video_name][pedId]
                    elif len(cv_split_inds) == 1:
                        db[video_name][pedId]['frames'] = cv_frame_list[0]
                        db[video_name][pedId]['cv_annotations']['bbox'] = cv_frame_box[0]
                        self.get_intent_des(db, video_name, pedId, cv_split_inds[0], cog_annotation)
                    else:
                        # multiple splits left after removing missing box frames
                        # print(f"{len(cv_frame_list)} splits: , {[len(s) for s in cv_frame_list]} \n")
                        nlp_vid_uid_pairs = db[video_name][pedId]['nlp_annotations'].keys()
                        for i in range(len(cv_frame_list)):
                            ped_splitId = pedId + '-' + str(i)
                            self.add_ped_case(db, video_name, ped_splitId, nlp_vid_uid_pairs)
                            db[video_name][ped_splitId]['frames'] = cv_frame_list[i]
                            db[video_name][ped_splitId]['cv_annotations']['bbox'] = cv_frame_box[i]
                            self.get_intent_des(db, video_name, ped_splitId, cv_split_inds[i], cog_annotation)
                            if len(db[video_name][ped_splitId]['nlp_annotations'][list(db[video_name][ped_splitId]['nlp_annotations'].keys())[0]]['intent']) == 0:
                                raise Exception("ERROR!")
                        del db[video_name][pedId] # no pedestrian list left, remove this video
                tracks.remove(pedId)

            # Object annotation
            for pedId in list(db[video_name].keys()):
                frames = db[video_name][pedId]['frames']
                if frames is None or len(frames) == 0:
                    continue
                for frame in frames:
                    frame = 'frame_' + str(frame)
                    if frame not in cv_annotation['frames']:
                        print(f"Frame, {frame}, not in cv_annotation! \n")
                        continue
                    obj_classes = []
                    obj_bboxes = []
                    obj_ids = []
                    for obj_id in cv_annotation['frames'][frame]['cv_annotation'].keys():
                        obj = cv_annotation['frames'][frame]['cv_annotation'][obj_id]
                        # Select only trained classes
                        if obj['object_type'] in self.object_class_list:
                            obj_classes.append(obj['object_type'])
                            obj_bboxes.append(obj['bbox'])
                            obj_ids.append(obj['track_id'])

                    db[video_name][pedId]['objs_classes'].append(obj_classes)
                    db[video_name][pedId]['objs_bboxes'].append(obj_bboxes)
                    db[video_name][pedId]['objs_ids'].append(obj_ids)


            if len(db[video_name].keys()) < 1: # has no valid ped sequence! Remove this video!")
                print(f"!!!!! Video, {video_name}, has no valid ped sequence! Remove this video! \n")
                del db[video_name]
            if len(tracks) > 0:
                print(f"{video_name} missing pedestrians annotations: {tracks}  \n")
    


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
            },
            'objs_classes': [],
            'objs_bboxes': [],
            'objs_ids': []
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

    def generate_data_sequence(self, set_name, database):
        intention_prob = []
        intention_binary = []
        frame_seq = []
        pids_seq = []
        video_seq = []
        box_seq = []
        description_seq = []
        disagree_score_seq = []
        objs_classes_seq = []
        objs_bboxes_seq = []    
        objs_ids_seq = []

        video_ids = sorted(database.keys())
        for video in sorted(video_ids): # video_name: e.g., 'video_0001'
            for ped in sorted(database[video].keys()): # ped_id: e.g., 'track_1'
                frame_seq.append(database[video][ped]['frames'])
                box_seq.append(database[video][ped]['cv_annotations']['bbox'])
                objs_classes_seq.append(database[video][ped]['objs_classes'])
                objs_bboxes_seq.append(database[video][ped]['objs_bboxes'])
                objs_ids_seq.append(database[video][ped]['objs_ids'])

                n = len(database[video][ped]['frames'])
                pids_seq.append([ped] * n)
                video_seq.append([video] * n)
                intents, probs, disgrs, descripts = self.get_intent(database, video, ped)
                intention_prob.append(probs)
                intention_binary.append(intents)
                disagree_score_seq.append(disgrs)
                description_seq.append(descripts)
            

        return {
            'frame': frame_seq,
            'bbox': box_seq,
            'objs_classes': objs_classes_seq,
            'objs_bboxes': objs_bboxes_seq,
            'objs_ids': objs_ids_seq,
            'intention_prob': intention_prob,
            'intention_binary': intention_binary,
            'ped_id': pids_seq,
            'video_id': video_seq,
            'disagree_score': disagree_score_seq,
            'description': description_seq
        }


    def get_intent(self, database, video_name, ped_id, intent_num = 3):
        prob_seq = []
        intent_seq = []
        disagree_seq = []
        description_seq = []
        n_frames = len(database[video_name][ped_id]['frames'])

        if self.intent_type == 'major' or self.intent_type == 'soft_vote':
            vid_uid_pairs = sorted((database[video_name][ped_id]['nlp_annotations'].keys()))
            n_users = len(vid_uid_pairs)
            for i in range(n_frames):
                labels = [database[video_name][ped_id]['nlp_annotations'][vid_uid]['intent'][i] for vid_uid in vid_uid_pairs]
                descriptions = [database[video_name][ped_id]['nlp_annotations'][vid_uid]['description'][i] for vid_uid in vid_uid_pairs]

                if intent_num == 3: # major 3 class, use cross-entropy loss
                    uni_lbls, uni_cnts = np.unique(labels, return_counts=True)
                    intent_binary = uni_lbls[np.argmax(uni_cnts)]
                    if intent_binary == 'not_cross':
                        intent_binary = 0
                    elif intent_binary == 'not_sure':
                        intent_binary = 1
                    elif intent_binary == 'cross':
                        intent_binary = 2
                    else:
                        raise Exception("ERROR intent label from database: ", intent_binary)

                    intent_prob = np.max(uni_cnts) / n_users
                    prob_seq.append(intent_prob)
                    intent_seq.append(intent_binary)
                    disagree_seq.append(1 - intent_prob)
                    description_seq.append(descriptions)
                elif intent_num == 2: # only counts labels not "not-sure", but will involve issues if all annotators are not-sure.
                    raise Exception("Sequence processing not implemented!")
                else:
                    pass
        elif self.intent_type == 'mean':
            vid_uid_pairs = sorted((database[video_name][ped_id]['nlp_annotations'].keys()))
            n_users = len(vid_uid_pairs)
            for i in range(n_frames):
                labels = [database[video_name][ped_id]['nlp_annotations'][vid_uid]['intent'][i] for vid_uid in vid_uid_pairs]
                descriptions = [database[video_name][ped_id]['nlp_annotations'][vid_uid]['description'][i] for vid_uid in vid_uid_pairs]

                for j in range(len(labels)):
                    if labels[j] == 'not_sure':
                        labels[j] = 0.5
                    elif labels[j] == 'not_cross':
                        labels[j] = 0
                    elif labels[j] == 'cross':
                        labels[j] = 1
                    else:
                        raise Exception("Unknown intent label: ", labels[j])
                # [0, 0.5, 1]
                intent_prob = np.mean(labels)
                intent_binary = 0 if intent_prob < 0.5 else 1
                prob_seq.append(intent_prob)
                intent_seq.append(intent_binary)
                disagree_score = sum([1 if lbl != intent_binary else 0 for lbl in labels]) / n_users
                disagree_seq.append(disagree_score)
                description_seq.append(descriptions)

        return intent_seq, prob_seq, disagree_seq, description_seq
