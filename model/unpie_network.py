from model.unpie_gcn import UnPIEGCN
from model.unpie_temporal_aggregator import UnPIETemporalAggregator
import numpy
import sys

from utils.graph_utils import graph_to_graph_seq
numpy.set_printoptions(threshold=sys.maxsize)
numpy.set_printoptions(linewidth=numpy.inf)

class UnPIENetwork(object):
    def __init__(self, middle_dim, emb_dim):
        self.gcn = UnPIEGCN(middle_dim, emb_dim)
        self.temporal_aggregator = UnPIETemporalAggregator(emb_dim)
    
    def __call__(self, x, a): 
        '''
        Args:
            x: tf.Tensor, shape=(batch_size, num_frames, num_nodes, num_channels + 4)
            a: tf.Tensor, shape=(batch_size, num_frames, num_nodes, num_nodes)
            y: tf.Tensor, shape=(batch_size)
        '''
        # Aggregate spatial features using a GNN
        ped_feat = self.gcn(x, a) # ped_feat shape: (batch_size, num_frames, emb_dim)

        # Aggregate temporal features
        ped_feat = self.temporal_aggregator(ped_feat) # ped_feat shape: (batch_size, emb_dim)
        return ped_feat