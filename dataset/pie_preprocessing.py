import os
import torch
import numpy as np
import pickle

from pathlib import PurePath

from dataset.pie_data import PIE
from dataset.pie_dataset import PIEGraphDataset
from model.unpie_gcn import UnPIEGCN
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
        self.inference_batch_size = params['inference_batch_size']
        self.img_height = params['img_height']
        self.img_width = params['img_width']
        self.edge_weigths = params['edge_weigths']
        self.edge_importance = params['edge_importance']
        self.feature_extractor = params['feature_extractor']
        self.data_sets = params['data_sets']

        self.ped_class = 'ped'
        self.other_ped_class = 'other_ped'

        self.pie = PIE(data_path=self.pie_path, data_opts=self.data_opts, data_sets=self.data_sets)

    def get_datasets(self):
        '''
        Build the inputs for the clustering computation
        '''
        # PIE preprocessing
        print_separator('PIE preprocessing')

        # Load the data
        test_features= None
        train_features = self._get_features('train')
        val_features = self._get_features('val')
        test_features = self._get_features('test')

        # Get the max number of nodes for each class across all splits
        if self.edge_importance:
            max_nodes_dict, max_num_nodes = self._get_max_nodes_dict(
                [train_features['max_nodes_dict'], val_features['max_nodes_dict'], test_features['max_nodes_dict']]
            )
        else:
            max_nodes_dict = {}
            max_num_nodes = max(train_features['max_num_nodes'], val_features['max_num_nodes'], test_features['max_num_nodes'])

        # One-hot classes
        one_hot_classes = self._get_one_hot_classes(
            [train_features['obj_classes'], val_features['obj_classes'], test_features['obj_classes']]
        )

        print('Max number of nodes in a graph: {}'.format(max_num_nodes))
        print('Classes max number of nodes: {}'.format(max_nodes_dict))

        # Load the data
        test_dataset = None
        test_len = 0
        train_dataloader, train_len = self._get_dataloader('train', train_features, max_nodes_dict, max_num_nodes, one_hot_classes)
        val_dataloader, val_len = self._get_dataloader('val', val_features, max_nodes_dict, max_num_nodes, one_hot_classes)
        test_dataset, test_len = self._get_dataloader('test', test_features, max_nodes_dict, max_num_nodes, one_hot_classes)

        return {
            'train': {
                'dataloader': train_dataloader,
                'len': train_len,
                'num_nodes': max_num_nodes
            },
            'val': {
                'dataloader': val_dataloader,
                'len': val_len,
                'num_nodes': max_num_nodes
            },
            'test': {
                'dataloader': test_dataset,
                'len': test_len,
                'num_nodes': max_num_nodes
            }
        }

    def _get_one_hot_classes(self, list_obj_classes):
        # Get classes
        classes_set = set(
            obj_class
            for split in list_obj_classes
            for seq in split
            for frame in seq
            for obj_class in frame
        )
        classes_set.add(self.ped_class)
        classes_set.add(self.other_ped_class)

        # One-hot encode classes
        classes_list = sorted(classes_set)  # Sorting ensures consistent order between different trainings
        class_to_index = {cls: i for i, cls in enumerate(classes_list)}

        # One-hot encoding
        one_hot_encoding = {
            cls: [1 if i == idx else 0 for i in range(len(classes_list))]
            for cls, idx in class_to_index.items()
        }

        return one_hot_encoding
    
    def _get_features(self, data_split):
        # Generate data sequences
        features_d = self.pie.generate_data_trajectory_sequence(data_split, **self.data_opts)

        # Generate data mini sequences
        seq_length = self.data_opts['max_size_observe']
        seq_ovelap_rate = self.data_opts['seq_overlap_rate']
        features_d = self._get_data(features_d, seq_length, seq_ovelap_rate)

        # Balance the number of samples in each split
        features_d = self.pie.balance_samples_count(features_d, label_type='intention_binary')

        # Load image features, train_img shape: (num_seqs, seq_length, embedding_size)
        features_d = self._load_features(features_d, data_split=data_split)

        return features_d

    def _get_dataloader(self, data_split, features_d, max_nodes_dict, max_num_nodes, one_hot_classes):
        pie_dataset = PIEGraphDataset(
            features_d,
            max_nodes_dict,
            max_num_nodes,
            one_hot_classes,
            [self.ped_class, self.other_ped_class],
            self.edge_importance,
            transform_a=UnPIEGCN.transform,
            height=self.img_height,
            width=self.img_width,
            edge_weigths=self.edge_weigths
        )

        if data_split == 'train':
            return torch.utils.data.DataLoader(pie_dataset, batch_size=self.batch_size, shuffle=True), pie_dataset.__len__()
        else:
            return torch.utils.data.DataLoader(pie_dataset, batch_size=self.inference_batch_size, shuffle=False), pie_dataset.__len__()
            
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
        obj_classes = dataset['obj_classes'].copy() # shape: (num_ped, num_frames, num_seq, num_objs, 1)
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
        obj_classes = self._get_tracks(obj_classes, seq_length, overlap_stride)
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
                'obj_classes': obj_classes,
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
        obj_classes = data['obj_classes']
        other_ped_ids = data['other_ped_ids']

        # load the feature files if exists
        peds_load_path = self.pie.get_path(
            data_type='features'+'_'+self.data_opts['crop_type']+'_'+self.data_opts['crop_mode'], # images    
            model_name=self.feature_extractor,
            data_subset = data_split,
            feature_type=self.pie.get_ped_type())
        objs_load_path = self.pie.get_path(
            data_type='features'+'_'+self.data_opts['crop_type']+'_'+self.data_opts['crop_mode'], # images    
            model_name=self.feature_extractor,
            data_subset = data_split,
            feature_type=self.pie.get_traffic_type())
        print("Loading {} features crop_type=context crop_mode=pad_resize \nsave_path={}, ".format(data_split, peds_load_path))

        ped_sequences, obj_sequences, other_ped_sequences = [], [], []
        max_class_nodes = {} # max number of nodes in a sequence, for padding
        max_num_nodes = 0
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            ped_seq, obj_seq, other_ped_seq = [], [], []
            for imp, p, o, o_c, op in zip(seq, pid, obj_ids[i], obj_classes[i], other_ped_ids[i]):
                # padding
                nodes_dict = {}
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
                nodes_dict[self.ped_class] = 1
                num_nodes += 1

                # object image features
                obj_seq_i = []
                img_save_folder = os.path.join(objs_load_path, set_id, vid_id)
                for obj_id, obj_class in zip(o, o_c):
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
                    nodes_dict[obj_class] = nodes_dict[obj_class] + 1 if obj_class in nodes_dict else 1
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
                    nodes_dict[self.other_ped_class] = nodes_dict[self.other_ped_class] + 1 if self.other_ped_class in nodes_dict else 1
                    num_nodes += 1
                other_ped_seq.append(other_ped_seq_i)

                # update max_nodes_dict
                max_class_nodes, _ = self._get_max_nodes_dict([max_class_nodes, nodes_dict])
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
            'obj_classes': data['obj_classes'],
            'other_ped_bboxes': data['other_ped_bboxes'],
            'intention_binary': data['intention_binary'], # shape: [num_seqs, 1]
            'data_split': data_split,
            'max_nodes_dict': max_class_nodes,
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
    
    def _get_max_nodes_dict(self, dict_list):
        # get the max number of nodes for each class in dict_list
        max_nodes_dict = {}
        for d in dict_list:
            for k, v in d.items():
                if k in max_nodes_dict:
                    max_nodes_dict[k] = max(max_nodes_dict[k], v)
                else:
                    max_nodes_dict[k] = v

        # count the number of nodes in the final dict
        max_num_nodes = 0
        for k, v in max_nodes_dict.items():
            max_num_nodes += v

        return max_nodes_dict, max_num_nodes 