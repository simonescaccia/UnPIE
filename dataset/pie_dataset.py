import torch
import numpy as np
import tensorflow as tf

from utils.pie_utils import bbox_center, update_progress

np.set_printoptions(linewidth=np.inf)

class PIEGraphDataset(torch.utils.data.Dataset):
    def __init__(self, features, max_num_nodes, height, width, transform_a=None):
        # self.x, self.a, self.i, self.y = self._compute_graphs(features, normalize_bbox, transform_a)
        self.features = features
        self.transform_a = transform_a
        self.max_num_nodes = max_num_nodes
        self.normalization_factor = np.linalg.norm([height, width])
        self.x, self.a, self.y, self.i = self._compute_graphs()


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
        max_num_nodes = self.max_num_nodes

        ped_feats = features['ped_feats'] # [num_seq, num_frames, emb_dim]
        ped_bboxes = features['ped_bboxes'] # [num_seq, num_frames, 4]
        ped_labels = features['intention_binary'] # [num_seq, 1]
        obj_feats = features['obj_feats'] # [num_seq, num_frames, num_obj, emb_dim]
        obj_bboxes = features['obj_bboxes'] # [num_seq, num_frames, num_obj, 4]
        other_ped_feats = features['other_ped_feats'] # [num_seq, num_frames, num_other_ped, emb_dim]
        other_ped_bboxes = features['other_ped_bboxes'] # [num_seq, num_frames, num_other_ped, 4]
        data_split = features['data_split']

        print('Computing {} graphs...'.format(data_split))

        num_seq = len(ped_feats)
        num_frames = len(ped_feats[0])

        # Pre-allocate for the sequence
        x_seq = np.zeros((num_seq, num_frames, max_num_nodes, ped_feats[0][0].shape[0]), dtype=np.float32)  
        a_seq = np.zeros((num_seq, num_frames, max_num_nodes, max_num_nodes), dtype=np.float32)
        y_seq = np.zeros((num_seq,), dtype=np.uint8)

        for i in range(num_seq):
            update_progress(i / num_seq)
            y_seq[i] = ped_labels[i]

            for j in range(num_frames):
                # Pedestrian node
                x_seq[i, j, 0, :] = ped_feats[i][j]
                ped_position = bbox_center(ped_bboxes[i][j])
                num_nodes = 1

                edge_weights = np.zeros((max_num_nodes,), dtype=np.float32) 

                # Object nodes
                for k, obj_feat in enumerate(obj_feats[i][j]):
                    x_seq[i, j, num_nodes, :] = obj_feat
                    edge_weights[num_nodes] = np.linalg.norm(np.array(ped_position) - np.array(bbox_center(obj_bboxes[i][j][k])))
                    num_nodes += 1
                
                # Other pedestrian nodes
                for k, other_ped_feat in enumerate(other_ped_feats[i][j]):
                    x_seq[i, j, num_nodes, :] = other_ped_feat
                    edge_weights[num_nodes] = np.linalg.norm(np.array(ped_position) - np.array(bbox_center(other_ped_bboxes[i][j][k])))
                    num_nodes += 1

                # Adjacency matrix: Copy the precomputed adjacency template and adjust for number of nodes
                # edge_weights = 1 - (edge_weights / self.normalization_factor) # Normalize edge weights
                a_seq[i, j, 0, :] = edge_weights
                a_seq[i, j, :, 0] = edge_weights

                if transform_a is not None:
                    a_seq[i, j, :, :] = transform_a(a_seq[i, j, :, :])

        update_progress(1)
        print('')

        x = x_seq
        a = a_seq
        y = y_seq
        i = np.arange(num_seq)

        return x, a, y, i

    def __getitem__(self, index):
        return self.x[index], self.a[index], self.y[index], self.i[index]
    
    def __len__(self):
        return self.x.shape[0]