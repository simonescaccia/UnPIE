import os
import numpy as np
import pickle5 as pickle
# import pickle (tensorflow 2)

from pathlib import PurePath

from dataset.pie_data import PIE
from dataset.pie_dataset import PIEGraphDataset
from model.unpie_gcn import UnPIEGCN
from utils.pie_utils import update_progress
from utils.print_utils import print_separator
from torch.utils.data import DataLoader


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

        # Generate data sequences
        train_d = self.pie.generate_data_trajectory_sequence('train', **self.data_opts)
        val_d = self.pie.generate_data_trajectory_sequence('val', **self.data_opts)
        if is_test:
            test_d = self.pie.generate_data_trajectory_sequence('test', **self.data_opts)

        # Generate data mini sequences
        seq_length = self.data_opts['max_size_observe']
        seq_ovelap_rate = self.data_opts['seq_overlap_rate']
        train_d = self._get_data(train_d, seq_length, seq_ovelap_rate)
        val_d = self._get_data(val_d, seq_length*self.val_num_clips, seq_ovelap_rate)
        if is_test:
            test_d = self._get_data(test_d, seq_length*self.test_num_clips, seq_ovelap_rate)

        # Balance the number of samples in each split
        train_d = self.pie.balance_samples_count(train_d, label_type='intention_binary')
        val_d = self.pie.balance_samples_count(val_d, label_type='intention_binary')

        # Load image features, train_img shape: (num_seqs, seq_length, embedding_size)
        train_d = self._load_features(train_d, data_split='train')
        val_d = self._load_features(val_d, data_split='val')
        if is_test:
            test_d = self._load_features(test_d, data_split='test')

        # Create dataloaders
        test_loader = None
        train_loader = self._get_dataloader(train_d)
        val_loader = self._get_dataloader(val_d)
        if is_test:
            test_loader = self._get_dataloader(test_d)

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

    def _get_dataloader(self, features):
        '''
        Create a dataloader for the clustering computation
        '''
        dataset = PIEGraphDataset(features, transform=UnPIEGCN.transform)
        return { # switch statement, python < 3.10 support
            'train':
                DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=True),
            'val':
                DataLoader(
                    dataset, batch_size=self.val_batch_size, shuffle=False),
            'test':
                DataLoader(
                    dataset, batch_size=self.test_batch_size, shuffle=False)
        }[features['data_split']]

        
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
        obj_bboxes = dataset['obj_bboxes'].copy() # shape: (num_ped, num_frames, num_seq, num_objs, 4)
        obj_ids = dataset['obj_ids'].copy() # shape: (num_ped, num_frames, num_seq, num_objs, 1)
        other_ped_bboxes = dataset['other_ped_bboxes'].copy() # shape: (num_ped, num_frames, num_seq, num_other_peds, 4)
        other_ped_ids = dataset['other_ped_ids'].copy() # shape: (num_ped, num_frames, num_seq, num_other_peds, 1)
        ped_ids = dataset['ped_ids'].copy() # shape: (num_ped, num_frames, 1)
        int_bin = dataset['intention_binary'].copy() # shape: (num_ped, num_frames, 1)

        overlap_stride = seq_length if overlap == 0 else \
        int((1 - overlap) * seq_length)

        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        bboxes = self._get_tracks(bboxes, seq_length, overlap_stride)
        images = self._get_tracks(images, seq_length, overlap_stride)
        obj_bboxes = self._get_tracks(obj_bboxes, seq_length, overlap_stride)
        obj_ids = self._get_tracks(obj_ids, seq_length, overlap_stride)
        other_ped_bboxes = self._get_tracks(other_ped_bboxes, seq_length, overlap_stride)
        other_ped_ids = self._get_tracks(other_ped_ids, seq_length, overlap_stride)
        ped_ids = self._get_tracks(ped_ids, seq_length, overlap_stride)
        int_bin = self._get_tracks(int_bin, seq_length, overlap_stride)

        bboxes = np.array(bboxes) # shape: (num_seqs, seq_length, 4)
        int_bin = np.array(int_bin)[:, 0] # every frame has the same intention label
        int_bin = np.squeeze(int_bin, axis=1) # shape: (num_seqs, 1)

        return {'images': images,
                'bboxes': bboxes,
                'obj_bboxes': obj_bboxes,
                'obj_ids': obj_ids,
                'other_ped_bboxes': other_ped_bboxes,
                'other_ped_ids': other_ped_ids,
                'ped_ids': ped_ids,
                'intention_binary': int_bin}

    def _load_features(self, data, data_split):
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
        img_sequences = data['images']
        ped_ids = data['ped_ids']
        obj_ids = data['obj_ids']
        other_ped_ids = data['other_ped_ids']

        # load the feature files if exists
        peds_load_path = self.pie.get_path(type_save='data',
                                    data_type='features'+'_'+self.data_opts['crop_type']+'_'+self.data_opts['crop_mode'], # images    
                                    model_name='vgg16_'+'none',
                                    data_subset = data_split,
                                    feature_type=self.pie.get_ped_type())
        objs_load_path = self.pie.get_path(type_save='data',
                                    data_type='features'+'_'+self.data_opts['crop_type']+'_'+self.data_opts['crop_mode'], # images    
                                    model_name='vgg16_'+'none',
                                    data_subset = data_split,
                                    feature_type=self.pie.get_traffic_type())
        print("Loading {} features crop_type=context crop_mode=pad_resize \nsave_path={}, ".format(data_split, peds_load_path))

        ped_sequences, obj_sequences, other_ped_sequences = [], [], []
        max_num_nodes = 0 # max number of nodes in a sequence, for padding
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            ped_seq, obj_seq, other_ped_seq = [], [], []
            for imp, p, o, op in zip(seq, pid, obj_ids[i], other_ped_ids[i]):
                num_nodes = 0
                set_id = PurePath(imp).parts[-3]
                vid_id = PurePath(imp).parts[-2]
                img_name = PurePath(imp).parts[-1].split('.')[0]

                # pedestrian image features
                img_save_folder = os.path.join(peds_load_path, set_id, vid_id)
                img_save_path = os.path.join(img_save_folder, img_name+'_'+p[0]+'.pkl')
                if not os.path.exists(img_save_path):
                    Exception("Image features not found at {}".format(img_save_path))
                with open(img_save_path, 'rb') as fid:
                    try:
                        img_features = pickle.load(fid)
                    except:
                        img_features = pickle.load(fid, encoding='bytes')
                img_features = np.squeeze(img_features) # VGG16 output
                ped_seq.append(img_features)
                num_nodes += 1

                # object image features
                obj_seq_i = []
                img_save_folder = os.path.join(objs_load_path, set_id, vid_id)
                for obj_id in o:
                    img_save_path = os.path.join(img_save_folder, img_name+'_'+obj_id+'.pkl')
                    if not os.path.exists(img_save_path):
                        Exception("Image features not found at {}".format(img_save_path))
                    with open(img_save_path, 'rb') as fid:
                        try:
                            img_features = pickle.load(fid)
                        except:
                            img_features = pickle.load(fid, encoding='bytes')
                    img_features = np.squeeze(img_features)
                    obj_seq_i.append(img_features)
                    num_nodes += 1
                obj_seq.append(obj_seq_i)

                # other pedestrian image features
                other_ped_seq_i = []
                img_save_folder = os.path.join(peds_load_path, set_id, vid_id)
                for op_id in op:
                    img_save_path = os.path.join(img_save_folder, img_name+'_'+op_id+'.pkl')
                    if not os.path.exists(img_save_path):
                        Exception("Image features not found at {}".format(img_save_path))
                    with open(img_save_path, 'rb') as fid:
                        try:
                            img_features = pickle.load(fid)
                        except:
                            img_features = pickle.load(fid, encoding='bytes')
                    img_features = np.squeeze(img_features)
                    other_ped_seq_i.append(img_features)
                    num_nodes += 1
                other_ped_seq.append(other_ped_seq_i)

                max_num_nodes = max(max_num_nodes, num_nodes)

            ped_sequences.append(ped_seq)
            obj_sequences.append(obj_seq)
            other_ped_sequences.append(other_ped_seq)
        update_progress(1)
        print("\n")
        features = {
            'ped_feats': ped_sequences,
            'obj_feats': obj_sequences,
            'other_ped_feats': other_ped_sequences,
            'ped_bboxes': data['bboxes'],
            'obj_bboxes': data['obj_bboxes'],
            'other_ped_bboxes': data['other_ped_bboxes'],
            'intention_binary': data['intention_binary'], # shape: [num_seqs, 1]
            'data_split': data_split,
            'max_num_nodes': max_num_nodes
        }
        return features

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