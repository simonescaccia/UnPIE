import numpy as np
from spektral.data import Dataset, Graph

from utils.graph_utils import graph_seq_to_graph
from utils.pie_utils import update_progress

class PIEGraphDataset(Dataset):
    def __init__(self, features, **kwargs):
        self.ped_feats = features['ped_feats'] # [num_seq, num_frames, emb_dim]
        self.ped_bboxes = features['ped_bboxes'] # [num_seq, num_frames, 4]
        self.ped_labels = features['intention_binary'] # [num_seq, num_frames]
        self.obj_feats = features['obj_feats'] # [num_seq, num_frames, num_obj, emb_dim]
        self.obj_bboxes = features['obj_bboxes'] # [num_seq, num_frames, num_obj, 4]
        self.other_ped_feats = features['other_ped_feats'] # [num_seq, num_frames, num_other_ped, emb_dim]
        self.other_ped_bboxes = features['other_ped_bboxes'] # [num_seq, num_frames, num_other_ped, 4]
        self.data_split = features['data_split']

        self.compute_graphs()
        super().__init__(**kwargs)

    def compute_graphs(self):
        '''
        Compute the graph structure of the dataset:
        - The pedestrian node is the center node
        - The object nodes are connected to the pedestrian node
        - The other pedestrian nodes are connected to the pedestrian node
        - The final graph is a star graph
        - The node features are the concatenation of the features and the bounding boxes
        '''
        print('\nComputing {} graphs...'.format(self.data_split))
        num_seq = len(self.ped_feats)
        num_frames = len(self.ped_feats[0])
        graph_seqs = []
        for i in range(num_seq):
            update_progress(i / num_seq)
            graph_seq = []
            y = self.ped_labels[i]
            y = np.array(y)
            num_nodes_seq = 0
            for j in range(num_frames):
                x = [] # node
                a = [] # adjacency matrix
                # Pedestrian node
                # TODO normalize the bounding box
                ped_node = np.concatenate((self.ped_feats[i][j], self.ped_bboxes[i][j]))
                x.append(ped_node)
                # Object nodes
                for k in range(len(self.obj_feats[i][j])):
                    obj_node = np.concatenate((self.obj_feats[i][j][k], self.obj_bboxes[i][j][k]))
                    x.append(obj_node)
                # Other pedestrian nodes
                for k in range(len(self.other_ped_feats[i][j])):
                    other_ped_node = np.concatenate((self.other_ped_feats[i][j][k], self.other_ped_bboxes[i][j][k]))
                    x.append(other_ped_node)
                # Adjacency matrix
                num_nodes = len(x)
                a.append([0] + [1] * (num_nodes-1))
                for k in range(num_nodes-1):
                    a.append([1] + [0] * (num_nodes-1))
                # Graph
                x = np.array(x)
                a = np.array(a)
                graph_seq.append(Graph(x=x, a=a, y=y))
                num_nodes_seq += num_nodes
            graph_seqs.append(graph_seq_to_graph(graph_seq, num_nodes_seq)) # spektral does not support graph sequence
        update_progress(1)
        print('')
        self.graphs = graph_seqs

    def read(self):
        return self.graphs