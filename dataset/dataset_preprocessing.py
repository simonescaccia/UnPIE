import os
import torch
import numpy as np
import pickle

from pathlib import PurePath

from dataset.pie_data import PIE
from dataset.graph_dataset import GraphDataset
from dataset.psi_data import PSI
from model.unpie_gcn import UnPIEGCN
from utils.data_utils import get_path, update_progress
from utils.print_utils import print_separator


class  DatasetPreprocessing(object):
    '''
    Given the PIE data, this class combines image features to create the image embedding for each pedestrian,
    generating input data for clustering computation.
    '''
    def __init__(self, params, dataset):
        self.data_opts = params['data_opts']
        self.batch_size = params['batch_size']
        self.inference_batch_size = params['inference_batch_size']
        self.img_height = params['img_height']
        self.img_width = params['img_width']
        self.edge_weigths = params['edge_weigths']
        self.feature_extractor = params['feature_extractor']
        self.data_sets = params['data_sets']
        self.balance_dataset = params['balance_dataset']
        self.feat_input_size = params['feat_input_size']
        self.obj_classes_list = params['obj_classes']

        self.path = 'dataset/ped_images.txt'
        # Delete the file if it exists
        if os.path.exists(self.path):
            # Delete the file
            os.remove(self.path)

        self.ped_class = 'ped'
        self.other_ped_class = 'other_ped'

        self.num_of_total_frames = 0
        self.dataset_objs_statistics = {obj: {} for obj in self.obj_classes_list}

        if dataset == 'pie':
            self.dataset_path = params['pie_path']
            self.dataset = PIE(
                data_path=self.dataset_path, 
                data_opts=self.data_opts, 
                data_sets=self.data_sets, 
                feat_input_size=self.feat_input_size, 
                feature_extractor=self.feature_extractor,
                obj_classes_list=self.obj_classes_list)
            self.dataset_name = 'PIE'
        elif dataset == 'psi':
            self.dataset_path = params['psi_path']
            self.dataset = PSI(
                data_path=self.dataset_path, 
                data_opts=self.data_opts,
                feat_input_size=self.feat_input_size, 
                feature_extractor=self.feature_extractor,
                obj_classes_list=self.obj_classes_list)
            self.dataset_name = 'PSI'  
        else:
            raise Exception("Unknown dataset name!")
        
    def get_datasets(self, is_only_test=False):
        '''
        Build the inputs for the clustering computation
        '''
        # PIE preprocessing
        print_separator('{} preprocessing'.format(self.dataset_name))

        # Load the test data
        test_features = self._get_features('test')

        if not is_only_test:
            train_features = self._get_features('train')
            val_features = self._get_features('val')
        else:
            train_features = val_features = None

        self._prints_dataset_statistics()

        # Print statistics: Get the max number of nodes for each class across all splits
        class_max_nodes_dict, _ = self._get_max_nodes_dict(
            [train_features['max_nodes_dict'], val_features['max_nodes_dict'], test_features['max_nodes_dict']]
            if not is_only_test else [test_features['max_nodes_dict']]
        )
        print('Classes max number of nodes: {}'.format(class_max_nodes_dict))

        graph_nodes_classes = self.obj_classes_list + [self.ped_class]
        one_hot_classes = self._get_one_hot_classes(graph_nodes_classes)

        print('Number of nodes in a graph: {}'.format(len(graph_nodes_classes)))

        # Load the test data
        test_dataset, test_len = self._get_dataloader('test', test_features, graph_nodes_classes, one_hot_classes)

        datasets = {
            'test': {
                'dataloader': test_dataset,
                'len': test_len
            }
        }

        if not is_only_test:
            train_dataloader, train_len = self._get_dataloader('train', train_features, graph_nodes_classes, one_hot_classes)
            val_dataloader, val_len = self._get_dataloader('val', val_features, graph_nodes_classes, one_hot_classes)

            datasets.update({
                'train': {
                    'dataloader': train_dataloader,
                    'len': train_len
                },
                'val': {
                    'dataloader': val_dataloader,
                    'len': val_len
                }
            })

        return datasets

    def _get_one_hot_classes(self, list_obj_classes):
        # One-hot encode classes
        classes_list = sorted(list_obj_classes)  # Sorting ensures consistent order between different trainings
        class_to_index = {cls: i for i, cls in enumerate(classes_list)}

        # One-hot encoding
        one_hot_encoding = {
            cls: [1 if i == idx else 0 for i in range(len(classes_list))]
            for cls, idx in class_to_index.items()
        }

        return one_hot_encoding
    
    def _get_features(self, data_split):
        # Generate data sequences
        features_d = self.dataset.generate_data_sequence(data_split, **self.data_opts)

        self._extract_dataset_statistics(features_d.copy())

        # Generate data mini sequences
        seq_length = self.data_opts['max_size_observe']
        seq_ovelap_rate = self.data_opts['seq_overlap_rate']
        features_d = self._get_data(features_d, seq_length, seq_ovelap_rate)

        # Balance the number of samples in each split
        if self.balance_dataset:
            features_d = self._balance_samples_count(features_d, label_type='intention_binary')

        # Save images for each ped_id from TP, TN, FP, FN ids
        for idx in range(len(features_d['ped_ids'])):
            ped_id = features_d['ped_ids'][idx]
            img_seq = features_d['images'][idx]
            with open(self.path, 'a') as f:
                for img in img_seq:
                    f.write(f"{ped_id[0][0]} {img} \n")

        # Load image features, train_img shape: (num_seqs, seq_length, embedding_size)
        features_d = self._load_features(features_d, data_split=data_split)

        return features_d

    def _get_dataloader(self, data_split, features_d, graph_nodes_classes, one_hot_classes):
        graph_dataset = GraphDataset(
            features_d,
            graph_nodes_classes,
            one_hot_classes,
            [self.ped_class, self.other_ped_class],
            transform_a=UnPIEGCN.transform,
            height=self.img_height,
            width=self.img_width,
            edge_weights=self.edge_weigths
        )

        if data_split == 'train':
            return torch.utils.data.DataLoader(graph_dataset, batch_size=self.batch_size, shuffle=True), graph_dataset.__len__()
        else:
            graph_dataset.shuffle()
            return torch.utils.data.DataLoader(graph_dataset, batch_size=self.inference_batch_size, shuffle=False), graph_dataset.__len__()
            
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
        peds_load_path = get_path(
            dataset_path=self.dataset_path,
            data_type='features'+'_'+self.data_opts['crop_type']+'_'+self.data_opts['crop_mode'], # images    
            model_name=self.feature_extractor,
            feature_type=self.dataset.get_ped_type())
        objs_load_path = get_path(
            dataset_path=self.dataset_path,
            data_type='features'+'_'+self.data_opts['crop_type']+'_'+self.data_opts['crop_mode'], # images    
            model_name=self.feature_extractor,
            feature_type=self.dataset.get_traffic_type())
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
    
    def _extract_dataset_statistics(self, dataset):
        # images = dataset['image'].copy() # shape: (num_ped, num_frames, channels)
        # bboxes = dataset['bbox'].copy() # shape: (num_ped, num_frames, 4)
        # obj_classes = dataset['obj_classes'].copy() # shape: (num_ped, num_frames, num_objs, obj)
        # obj_bboxes = dataset['obj_bboxes'].copy() # shape: (num_ped, num_frames, num_objs, 4)
        # obj_ids = dataset['obj_ids'].copy() # shape: (num_ped, num_frames, num_objs, obj)
        # other_ped_bboxes = dataset['other_ped_bboxes'].copy() # shape: (num_ped, num_frames, num_other_peds, 4)
        # other_ped_ids = dataset['other_ped_ids'].copy() # shape: (num_ped, num_frames, num_other_peds)
        # ped_ids = dataset['ped_ids'].copy() # shape: (num_ped, num_frames)
        # int_bin = dataset['intention_binary'].copy() # shape: (num_ped, num_frames)
        for objs_seq_frames, other_peds_seq_frames in zip(dataset['obj_classes'], dataset['other_ped_ids']):
            self.num_of_total_frames += len(objs_seq_frames)

            for frame_objs, frame_other_peds in zip(objs_seq_frames, other_peds_seq_frames):
                frame_objs_dict = {obj: 0 for obj in self.obj_classes_list}
                # Objects
                for obj in frame_objs:
                    if obj not in frame_objs_dict.keys(): # obj not found on the main dict
                        continue
                    idx = frame_objs_dict[obj]
                    if idx not in self.dataset_objs_statistics[obj].keys(): # obj index not found on the main dict
                        self.dataset_objs_statistics[obj][idx] = 1
                    else:
                        self.dataset_objs_statistics[obj][idx] += 1
                    frame_objs_dict[obj] += 1
                
                # Other peds
                if 'other_ped' not in frame_objs_dict.keys(): # other ped not found on the main dict
                    continue
                for i, _ in enumerate(frame_other_peds):
                    if i not in self.dataset_objs_statistics['other_ped'].keys(): # other ped index not found on the main dict
                        self.dataset_objs_statistics['other_ped'][i] = 1
                    else:
                        self.dataset_objs_statistics['other_ped'][i] += 1

    def _prints_dataset_statistics(self):
        # Get the maximum number of nodes within the same class
        num_classes = len(self.dataset_objs_statistics)
        max_num_nodes = 0
        for key in self.dataset_objs_statistics.keys():
            max_num_nodes = max(max_num_nodes, len(self.dataset_objs_statistics[key]))

        # Create a table
        table = np.zeros((num_classes, max_num_nodes), dtype=np.float32)
        rows = []
        for i, obj_key in enumerate(self.dataset_objs_statistics.keys()):
            rows.append(obj_key)
            for j, value in self.dataset_objs_statistics[obj_key].items():
                table[i, j] = value
        
        # Compute the frequency of each occurrence
        table = np.round(table / self.num_of_total_frames, 2)

        # Compute the mean per column
        mean_occurrence = np.round(np.mean(table, axis=0), 2)

        # Print table
        print("Table dataset objects")
        print("rows: ", rows)
        print(table)
        print("Mean occurrence", mean_occurrence)

    def _balance_samples_count(self, seq_data, label_type, random_seed=42):
        """
        Balances the number of positive and negative samples by randomly sampling
        from the more represented samples. Only works for binary classes.
        :param seq_data: The sequence data to be balanced.
        :param label_type: The lable type based on which the balancing takes place.
        The label values must be binary, i.e. only 0, 1.
        :param random_seed: The seed for random number generator.
        :return: Balanced data sequence.
        """
        for lbl in seq_data[label_type]:
            if lbl not in [0, 1]:
                raise Exception("The label values used for balancing must be"
                                " either 0 or 1")

        # balances the number of positive and negative samples
        print_separator("Balancing the number of positive and negative intention samples", space=False)

        gt_labels = seq_data[label_type]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        new_seq_data = {}
        # finds the indices of the samples with larger quantity
        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
            return seq_data
        else:
            print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
            if num_neg_samples > num_pos_samples:
                rm_index = np.where(np.array(gt_labels) == 0)[0]
            else:
                rm_index = np.where(np.array(gt_labels) == 1)[0]

            # Calculate the difference of sample counts
            dif_samples = abs(num_neg_samples - num_pos_samples)
            # shuffle the indices
            np.random.seed(random_seed)
            np.random.shuffle(rm_index)
            # reduce the number of indices to the difference
            rm_index = rm_index[0:dif_samples]
            # update the data
            for k in seq_data:
                seq_data_k = seq_data[k]
                if not isinstance(seq_data[k], list) and not isinstance(seq_data[k], np.ndarray):
                    new_seq_data[k] = seq_data[k]
                else:
                    new_seq_data[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

            new_gt_labels = new_seq_data[label_type]
            num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
            print('Balanced:\t Positive: %d  \t Negative: %d\n'
                  % (num_pos_samples, len(new_seq_data[label_type]) - num_pos_samples))
        return new_seq_data
                

