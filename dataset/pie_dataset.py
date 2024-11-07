import numpy as np
import tensorflow as tf

from utils.pie_utils import update_progress

class PIEGraphDataset():
    def __init__(self, features, normalize_bbox, batch_size, shuffle, transform_a=None):
        self.dataset = self._compute_graphs(features, normalize_bbox, batch_size, shuffle, transform_a)

    def _compute_graphs(self, features, normalize_bbox, batch_size, shuffle, transform_a):
        '''
        Compute the graph structure of the dataset:
        - The pedestrian node is the center node
        - The object nodes are connected to the pedestrian node
        - The other pedestrian nodes are connected to the pedestrian node
        - The final graph is a star graph
        - The node features are the concatenation of the features and the bounding boxes
        '''
        ped_feats = features['ped_feats'] # [num_seq, num_frames, emb_dim]
        ped_bboxes = features['ped_bboxes'] # [num_seq, num_frames, 4]
        ped_labels = features['intention_binary'] # [num_seq, 1]
        obj_feats = features['obj_feats'] # [num_seq, num_frames, num_obj, emb_dim]
        obj_bboxes = features['obj_bboxes'] # [num_seq, num_frames, num_obj, 4]
        other_ped_feats = features['other_ped_feats'] # [num_seq, num_frames, num_other_ped, emb_dim]
        other_ped_bboxes = features['other_ped_bboxes'] # [num_seq, num_frames, num_other_ped, 4]
        data_split = features['data_split']
        max_num_nodes = features['max_num_nodes']

        print('\nComputing {} graphs...'.format(data_split))

        num_seq = len(ped_feats)
        num_frames = len(ped_feats[0])
        
         # Precompute number of nodes and adjacency template to reduce redundant calculations
        adj_template = np.zeros((max_num_nodes, max_num_nodes), dtype=int)
        adj_template[0, 1:max_num_nodes] = 1  # Central node connected to others
        adj_template[1:max_num_nodes, 0] = 1  # Other nodes connected to central node

        # Pre-allocate for the sequence
        x_seq = np.zeros((num_seq, num_frames, max_num_nodes, ped_feats[0][0].shape[0] + ped_bboxes[0][0].shape[0]), dtype=np.float32)  
        a_seq = np.zeros((num_seq, num_frames, max_num_nodes, max_num_nodes), dtype=np.uint8)
        y_seq = np.zeros((num_seq,), dtype=np.uint8)

        for i in range(num_seq):
            update_progress(i / num_seq)
            y_seq[i] = ped_labels[i]

            for j in range(num_frames):
                # Pedestrian node
                num_nodes = 1
                ped_node = np.concatenate((ped_feats[i][j], normalize_bbox(ped_bboxes[i][j])))
                x_seq[i, j, 0, :] = ped_node

                # Object nodes
                for k, obj_feat in enumerate(obj_feats[i][j]):
                    obj_node = np.concatenate((obj_feat, normalize_bbox(obj_bboxes[i][j][k])))
                    x_seq[i, j, num_nodes, :] = obj_node
                    num_nodes += 1
                
                # Other pedestrian nodes
                for k, other_ped_feat in enumerate(other_ped_feats[i][j]):
                    other_ped_node = np.concatenate((other_ped_feat, normalize_bbox(other_ped_bboxes[i][j][k])))
                    x_seq[i, j, num_nodes, :] = other_ped_node
                    num_nodes += 1

                # Adjacency matrix: Copy the precomputed adjacency template and adjust for number of nodes
                a_seq[i, j, :, :] = adj_template
                if num_nodes < max_num_nodes:
                    a_seq[i, j, num_nodes:, :] = 0  # Zero out rows and columns beyond the active node count
                    a_seq[i, j, :, num_nodes:] = 0  # Zero out rows and columns beyond the active node count
                    if transform_a is not None:
                        a_seq[i, j, :, :] = transform_a(a_seq[i, j, :, :])

        update_progress(1)
        print('')

        self.x = x_seq
        self.a = a_seq
        self.y = y_seq
        self.i = np.arange(num_seq)

        dataset = tf.data.Dataset.from_tensor_slices(
            dict(x=self.x, a=self.a, y=self.y, i=self.i)
        )

        if shuffle:
            return dataset.shuffle(buffer_size=num_seq).batch(batch_size=batch_size)
        else:
            return dataset.batch(batch_size=batch_size)

    def get_dataset(self):
        return self.dataset