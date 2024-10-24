import numpy as np

from torch.utils.data import Dataset
from utils.pie_utils import update_progress
from scipy.sparse import csr_matrix

class PIEGraphDataset(Dataset):
    def __init__(self, features):
        self.x, self.a = self.compute_graphs(features)

    # def compute_graphs(self):
    #     '''
    #     Compute the graph structure of the dataset:
    #     - The pedestrian node is the center node
    #     - The object nodes are connected to the pedestrian node
    #     - The other pedestrian nodes are connected to the pedestrian node
    #     - The final graph is a star graph
    #     - The node features are the concatenation of the features and the bounding boxes
    #     '''
    #     print('\nComputing {} graphs...'.format(self.data_split))
    #     num_seq = len(self.ped_feats)
    #     num_frames = len(self.ped_feats[0])
    #     graph_seqs = []
    #     for i in range(num_seq):
    #         update_progress(i / num_seq)
    #         graph_seq = []
    #         y = self.ped_labels[i]
    #         y = np.array(y)
    #         num_nodes_seq = 0
    #         for j in range(num_frames):
    #             x = [] # node
    #             a = [] # adjacency matrix
    #             # Pedestrian node
    #             # TODO normalize the bounding box
    #             ped_node = np.concatenate((self.ped_feats[i][j], self.ped_bboxes[i][j]))
    #             x.append(ped_node)
    #             # Object nodes
    #             for k in range(len(self.obj_feats[i][j])):
    #                 obj_node = np.concatenate((self.obj_feats[i][j][k], self.obj_bboxes[i][j][k]))
    #                 x.append(obj_node)
    #             # Other pedestrian nodes
    #             for k in range(len(self.other_ped_feats[i][j])):
    #                 other_ped_node = np.concatenate((self.other_ped_feats[i][j][k], self.other_ped_bboxes[i][j][k]))
    #                 x.append(other_ped_node)
    #             # Adjacency matrix
    #             num_nodes = len(x)
    #             a.append([0] + [1] * (num_nodes-1))
    #             for k in range(num_nodes-1):
    #                 a.append([1] + [0] * (num_nodes-1))
    #             # Graph
    #             x = np.array(x)
    #             a = np.array(a)
    #             graph_seq.append(Graph(x=x, a=a, y=y))
    #             num_nodes_seq += num_nodes
    #         graph_seqs.append(graph_seq_to_graph(graph_seq, num_nodes_seq)) # spektral does not support graph sequence
    #     update_progress(1)
    #     print('')
    #     self.graphs = graph_seqs

    def compute_graphs(self, features):
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
        ped_labels = features['intention_binary'] # [num_seq, num_frames]
        obj_feats = features['obj_feats'] # [num_seq, num_frames, num_obj, emb_dim]
        obj_bboxes = features['obj_bboxes'] # [num_seq, num_frames, num_obj, 4]
        other_ped_feats = features['other_ped_feats'] # [num_seq, num_frames, num_other_ped, emb_dim]
        other_ped_bboxes = features['other_ped_bboxes'] # [num_seq, num_frames, num_other_ped, 4]
        data_split = features['data_split']
        max_num_nodes = features['max_num_nodes']

        print('\nComputing {} graphs...'.format(data_split))

        num_seq = len(ped_feats)
        num_frames = len(ped_feats[0])
        
        x_seqs = []
        a_seqs = []
        y_seqs = []
        
         # Precompute number of nodes and adjacency template to reduce redundant calculations
        adj_template = np.zeros((max_num_nodes, max_num_nodes), dtype=int)
        adj_template[0, 1:max_num_nodes] = 1  # Central node connected to others
        adj_template[1:max_num_nodes, 0] = 1  # Other nodes connected to central node

        # Pre-allocate for the sequence
        x_seq = np.zeros((num_seq, num_frames, max_num_nodes, ped_feats[0][0].shape[0] + ped_bboxes[0][0].shape[0]), dtype=np.float32)  
        a_seq = np.zeros((num_seq, num_frames, max_num_nodes, max_num_nodes), dtype=np.uint8)
        y_seq = np.zeros((num_seq, num_frames), dtype=np.uint8)

        for i in range(num_seq):
            update_progress(i / num_seq)
        
            y_seq[i, :] = ped_labels[i]

            for j in range(num_frames):
                # Pedestrian node
                num_nodes = 1
                ped_node = np.concatenate((ped_feats[i][j], ped_bboxes[i][j])) # TODO normalize the bounding box
                x_seq[i, j, 0, :] = ped_node

                # Object nodes
                for k, obj_feat in enumerate(obj_feats[i][j]):
                    obj_node = np.concatenate((obj_feat, obj_bboxes[i][j][k]))
                    x_seq[i, j, num_nodes, :] = obj_node
                    num_nodes += 1
                
                # Other pedestrian nodes
                for k, other_ped_feat in enumerate(other_ped_feats[i][j]):
                    other_ped_node = np.concatenate((other_ped_feat, other_ped_bboxes[i][j][k]))
                    x_seq[i, j, num_nodes, :] = other_ped_node
                    num_nodes += 1

                # Adjacency matrix: Copy the precomputed adjacency template and adjust for number of nodes
                a_seq[i, j, :, :] = adj_template
                if num_nodes < max_num_nodes:
                    a_seq[i, j, num_nodes:, :] = 0  # Zero out rows and columns beyond the active node count
                    a_seq[i, j, :, num_nodes:] = 0  # Zero out rows and columns beyond the active node count

        update_progress(1)
        print('')

        return x_seqs, a_seqs, y_seqs

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.a[idx], self.y[idx]