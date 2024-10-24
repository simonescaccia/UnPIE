from model.unpie_gcn import GCN
from model.unpie_temporal_aggregator import TemporalAggregator
import numpy
import sys

from utils.graph_utils import graph_to_graph_seq
numpy.set_printoptions(threshold=sys.maxsize)
numpy.set_printoptions(linewidth=numpy.inf)

class UnPIENetwork(object):
    def __init__(self, middle_dim, emb_dim):
        self.gcn = GCN(middle_dim, emb_dim)
        self.temporal_aggregator = TemporalAggregator(emb_dim)
    
    def __call__(self, x, a): 
        '''
        Args:
            x: tf.Tensor, shape=(batch_size, num_nodes, num_channels + 4)
            a: tf.Tensor, shape=(batch_size, num_nodes, num_nodes)
            y: tf.Tensor, shape=(batch_size)
        '''
        print('a\n', a) # TODO remove padding
        # Aggregate spatial features using a GNN
        ped_feat = []
        graph_seq = graph_to_graph_seq(x, a)
        for x, a in graph_seq:
            ped_feat.append(self.gcn(x, a))

        # Aggregate temporal features
        ped_feat = self.temporal_aggregator(ped_feat)
        return ped_feat