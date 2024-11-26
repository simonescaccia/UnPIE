import torch
import numpy as np
import tensorflow as tf

from utils.pie_utils import bbox_center, update_progress

np.set_printoptions(linewidth=np.inf)

class PIEGraphDataset(torch.utils.data.Dataset):
    def __init__(self, features, max_nodes_dict, max_num_nodes, ped_classes, edge_importance, height, width, edge_weigths, transform_a=None):
        self.edge_weigths = edge_weigths
        self.features = features
        self.transform_a = transform_a
        self.max_nodes_dict = max_nodes_dict
        self.max_num_nodes = max_num_nodes
        self.ped_classes = ped_classes
        self.edge_importance = edge_importance
        self.normalization_factor = np.linalg.norm([height, width])
        self.x, self.b, self.a, self.y, self.i = self._compute_graphs()


    def _compute_graphs(self):
        '''
        Compute the graph structure of the dataset:
        - The pedestrian node is the center node
        - The object nodes are connected to the pedestrian node
        - The other pedestrian nodes are connected to the pedestrian node
        - The final graph is a star graph
        - The node features are the concatenation of the features and the bounding boxes
        '''
        features = self.features
        transform_a = self.transform_a
        max_nodes_dict = self.max_nodes_dict
        max_num_nodes = self.max_num_nodes
        ped_classes = self.ped_classes
        edge_importance = self.edge_importance

        ped_feats = features['ped_feats'] # [num_seqs, seq_len, emb_dim]
        ped_bboxes = features['ped_bboxes'] # [num_seqs, seq_len, 4]
        ped_labels = features['intention_binary'] # [num_seqs, 1]
        obj_feats = features['obj_feats'] # [num_seqs, seq_len, num_obj, emb_dim]
        obj_bboxes = features['obj_bboxes'] # [num_seqs, seq_len, num_obj, 4]
        obj_classes = features['obj_classes'] # [num_seqs, seq_len, num_obj, str_len]
        other_ped_feats = features['other_ped_feats'] # [num_seqs, seq_len, num_other_ped, emb_dim]
        other_ped_bboxes = features['other_ped_bboxes'] # [num_seqs, seq_len, num_other_ped, 4]
        data_split = features['data_split']

        print('Computing {} graphs...'.format(data_split))

        # Prepare objects start position. Class order: ped, obj_class_1, obj_class_2, ..., obj_class_n , other_peds
        if edge_importance:
            class_init_pos = 1
            obj_class_pos = {}
            for k, v in max_nodes_dict.items():
                if k not in ped_classes:
                    obj_class_pos[k] = {}
                    obj_class_pos[k]['init_pos'] = class_init_pos
                    obj_class_pos[k]['count'] = 0
                    class_init_pos += v
            other_ped_init_pos = class_init_pos

        num_seqs = len(ped_feats)
        seq_len = len(ped_feats[0])

        # Pre-allocate for the sequence
        x_seq = np.zeros((num_seqs, seq_len, max_num_nodes, ped_feats[0][0].shape[0]), dtype=np.float32)
        b_seq = np.zeros((num_seqs, seq_len, max_num_nodes, 4), dtype=np.float32)
        a_seq = np.zeros((num_seqs, seq_len, max_num_nodes, max_num_nodes), dtype=np.float32)
        y_seq = np.zeros((num_seqs,), dtype=np.uint8)

        for i in range(num_seqs):
            update_progress(i / num_seqs)
            y_seq[i] = ped_labels[i]

            for j in range(seq_len):
                # Pedestrian node
                x_seq[i, j, 0, :] = ped_feats[i][j]
                b_seq[i, j, 0, :] = ped_bboxes[i][j]
                ped_position = bbox_center(ped_bboxes[i][j])
                num_node = 1

                edge_weights = np.zeros((max_num_nodes,), dtype=np.float32) 

                # Object nodes
                for k, (obj_feat, obj_bbox, obj_class) in enumerate(zip(obj_feats[i][j], obj_bboxes[i][j], obj_classes[i][j])):
                    if edge_importance:
                        num_node = obj_class_pos[obj_class]['init_pos'] + obj_class_pos[obj_class]['count']

                    x_seq[i, j, num_node, :] = obj_feat
                    b_seq[i, j, num_node, :] = obj_bbox
                    
                    if not self.edge_weigths:
                        edge_weights[num_node] = 1
                    else:
                        edge_weights[num_node] = self._get_distance(ped_position, bbox_center(obj_bboxes[i][j][k]))
                    
                    if edge_importance:
                        obj_class_pos[obj_class]['count'] += 1
                    else:
                        num_node += 1

                if edge_importance:
                    # Restore the object class position
                    for k, v in obj_class_pos.items():
                        obj_class_pos[k]['count'] = 0

                # Other pedestrian nodes
                if edge_importance:
                    num_node = other_ped_init_pos
                for k, (other_ped_feat, other_ped_bbox) in enumerate(zip(other_ped_feats[i][j], other_ped_bboxes[i][j])):
                    x_seq[i, j, num_node, :] = other_ped_feat
                    b_seq[i, j, num_node, :] = other_ped_bbox
                    if not self.edge_weigths:
                        edge_weights[num_node] = 1
                    else:    
                        edge_weights[num_node] = self._get_distance(ped_position, bbox_center(other_ped_bboxes[i][j][k]))
                    num_node += 1

                # Adjacency matrix: Copy the precomputed adjacency template and adjust for number of nodes
                a_seq[i, j, 0, :] = edge_weights
                a_seq[i, j, :, 0] = edge_weights

                print(a_seq[i, j, :, :])

                if transform_a is not None:
                    a_seq[i, j, :, :] = transform_a(a_seq[i, j, :, :])

        update_progress(1)
        print('')

        x = x_seq
        b = b_seq
        a = a_seq
        y = y_seq
        i = np.arange(num_seqs)

        return x, b, a, y, i
    
    def _get_distance(self, ped_position, obj_position):
        distance = np.linalg.norm(np.array(ped_position) - np.array(obj_position))
        if self.edge_weigths == 'no_norm':
            pass
        elif self.edge_weigths == 'norm':
            distance = distance / self.normalization_factor
        elif self.edge_weigths == 'compl':
            distance = self.normalization_factor - distance
        elif self.edge_weigths == 'norm_compl':
            distance = 1 - (distance / self.normalization_factor)
        return distance

    def __getitem__(self, index):
        return self.x[index], self.b[index], self.a[index], self.y[index], self.i[index]
    
    def __len__(self):
        return self.x.shape[0]