import os
import numpy as np
import pickle5 as pickle
# import pickle (tensorflow 2)

from pathlib import PurePath

import torch
from dataset.pie_data import PIE
from dataset.pie_dataset import PIEDataset
from utils.pie_utils import update_progress
from utils.print_utils import print_separator


class PIEPreprocessing(object):
    '''
    Given the PIE data, this class combines image features to create the image embedding for each pedestrian,
    generating input data for clustering computation.
    '''
    def __init__(self, params):
        self.pie_path = params['pie_path']
        self.data_opts = params['data_opts']
        self.batch_size = params['batch_size']
        self.val_batch_size = params['val_batch_size']
        self.val_num_clips = params['val_num_clips']
        self.test_batch_size = params['test_batch_size']
        self.test_num_clips = params['test_num_clips']

        self.pie = PIE(data_path=self.pie_path)

    def get_data_loaders(self, is_test):
        '''
        Build the inputs for the clustering computation
        '''
        # PIE preprocessing
        print_separator('PIE preprocessing',bottom_new_line=False)

        # Generate image sequences
        seq_train = self.pie.generate_data_trajectory_sequence('train', **self.data_opts)
        seq_val = self.pie.generate_data_trajectory_sequence('val', **self.data_opts)
        if is_test:
            seq_test = self.pie.generate_data_trajectory_sequence('test', **self.data_opts)

        seq_length = self.data_opts['max_size_observe']
        seq_ovelap_rate = self.data_opts['seq_overlap_rate']
        train_d = self._get_data(seq_train, seq_length, seq_ovelap_rate)
        val_d = self._get_data(seq_val, seq_length*self.val_num_clips, seq_ovelap_rate)
        if is_test:
            test_d = self._get_data(seq_test, seq_length*self.test_num_clips, seq_ovelap_rate)

        train_d = self.pie.balance_samples_count(train_d, label_type='intention_binary')
        val_d = self.pie.balance_samples_count(val_d, label_type='intention_binary')

        # Load image features, train_img shape: (num_seqs, seq_length, embedding_size)
        train_img = self._load_features(train_d['images'],
                                        train_d['ped_ids'],
                                        data_type='train',
                                        feature_type=self.pie.get_ped_type())
                                        
        val_img = self._load_features(val_d['images'],
                                      val_d['ped_ids'],
                                      data_type='val',
                                      feature_type=self.pie.get_ped_type())
        if is_test:
            test_img = self._load_features(test_d['images'],
                                           test_d['ped_ids'],
                                           data_type='test',
                                           feature_type=self.pie.get_ped_type())

        # Create dataloaders
        test_loader = None
        train_loader = self._get_dataloader(train_img, train_d['bboxes'], train_d['intention_binary'], 'train') # train_d['output'] shape: (num_seqs, 1)
        val_loader = self._get_dataloader(val_img, val_d['bboxes'], val_d['intention_binary'], 'val')
        if is_test:
            test_loader = self._get_dataloader(test_img, test_d['bboxes'], test_d['intention_binary'], 'test')

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

    def _get_dataloader(self, img_features, bboxes, labels, type):
        '''
        Create a dataloader for the clustering computation
        '''
        dataset = PIEDataset(img_features, bboxes, labels)
        return { # switch statement, python < 3.10 support
            'train':
                torch.utils.data.DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False),
            'val':
                torch.utils.data.DataLoader(
                    dataset, batch_size=self.val_batch_size, shuffle=False, pin_memory=False),
            'test':
                torch.utils.data.DataLoader(
                    dataset, batch_size=self.test_batch_size, shuffle=False, pin_memory=False)
        }[type]

        
    def _get_data(self, dataset, seq_length, overlap):
        """
        A helper function for data generation that combines different data types into a single
        representation.
        :param data: A dictionary of data types
        :param seq_length: the length of the sequence
        :param overlap: defines the overlap between consecutive sequences (between 0 and 1)
        :return: A unified data representation as a list.
        """
        images = dataset['image'].copy() # shape: (num_ped, num_frames, num_seq)
        bboxes = dataset['bbox'].copy() # shape: (num_ped, num_frames, num_seq, 4)
        objs_bboxes = dataset['objs_bbox'].copy() # shape: (num_ped, num_frames, num_seq, num_objs, 4)
        other_peds_bboxes = dataset['other_peds_bbox'].copy() # shape: (num_ped, num_frames, num_seq, num_other_peds, 4)
        ped_ids = dataset['ped_id'].copy() # shape: (num_ped, num_frames, 1)
        int_bin = dataset['intention_binary'].copy() # shape: (num_ped, num_frames, 1)

        overlap_stride = seq_length if overlap == 0 else \
        int((1 - overlap) * seq_length)

        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        bboxes = self._get_tracks(bboxes, seq_length, overlap_stride)
        images = self._get_tracks(images, seq_length, overlap_stride)
        objs_bboxes = self._get_tracks(objs_bboxes, seq_length, overlap_stride)
        other_peds_bboxes = self._get_tracks(other_peds_bboxes, seq_length, overlap_stride)
        ped_ids = self._get_tracks(ped_ids, seq_length, overlap_stride)
        int_bin = self._get_tracks(int_bin, seq_length, overlap_stride)

        bboxes = np.array(bboxes) # shape: (num_seqs, seq_length, 4)
        objs_bboxes = np.array(objs_bboxes) # shape: (num_seqs, seq_length, num_objs, 4)
        other_peds_bboxes = np.array(other_peds_bboxes) # shape: (num_seqs, seq_length, num_other_peds, 4)
        int_bin = np.array(int_bin)[:, 0] # every frame has the same intention label
        int_bin = np.squeeze(int_bin, axis=1) # shape: (num_seqs, 1)

        return {'images': images,
                'bboxes': bboxes,
                'objs_bboxes': objs_bboxes,
                'other_peds_bboxes': other_peds_bboxes,
                'ped_ids': ped_ids,
                'intention_binary': int_bin}

    def _load_features(self,
                      img_sequences,
                      ped_ids,
                      data_type,
                      feature_type):
        """
        Load image features. The images are first
        cropped to 1.5x the size of the bounding box, padded and resized to
        (224, 224) and fed into pretrained VGG16.
        :param img_sequences: a list of frame names
        :param bbox_sequences: a list of corresponding bounding boxes
        :ped_ids: a list of pedestrian ids associated with the sequences
        :save_path: path to save the precomputed features
        :data_type: train/val/test data set
        :regen_pkl: if set to True overwrites previously saved features
        :return: a list of image features
        """
        # load the feature files if exists
        load_path = self.pie.get_path(type_save='data',
                                    data_type='features'+'_'+self.data_opts['crop_type']+'_'+self.data_opts['crop_mode'], # images    
                                    model_name='vgg16_'+'none',
                                    data_subset = 'train',
                                    feature_type=feature_type)
        print("Loading {} features crop_type=context crop_mode=pad_resize \nsave_path={}, ".format(data_type, load_path))

        sequences = []
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            img_seq = []
            for imp, p in zip(seq, pid):
                set_id = PurePath(imp).parts[-3]
                vid_id = PurePath(imp).parts[-2]
                img_name = PurePath(imp).parts[-1].split('.')[0]
                img_save_folder = os.path.join(load_path, set_id, vid_id)
                img_save_path = os.path.join(img_save_folder, img_name+'_'+p[0]+'.pkl')
                if not os.path.exists(img_save_path):
                    Exception("Image features not found at {}".format(img_save_path))
                with open(img_save_path, 'rb') as fid:
                    try:
                        img_features = pickle.load(fid)
                    except:
                        img_features = pickle.load(fid, encoding='bytes')
                img_features = np.squeeze(img_features) # VGG16 output shape: (7, 7, 512)
                img_seq.append(img_features)
            sequences.append(img_seq)
        update_progress(1)
        print("\n")
        sequences = np.array(sequences)
        return sequences

    def _get_tracks(self, sequences, seq_length, overlap_stride):
        """
        Generate tracks by sampling from pedestrian sequences
        :param dataset: raw data from the dataset
        """
        sub_seqs = []
        for seq in sequences:
            sub_seqs.extend([seq[i:i+seq_length] for i in\
                         range(0,len(seq)\
                        - seq_length + 1, overlap_stride)])
        return sub_seqs