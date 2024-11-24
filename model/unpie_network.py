import tensorflow as tf


from model.unpie_st_gcn import UnPIESTGCN
from model.unpie_temporal_aggregator import UnPIETemporalAggregator

class UnPIENetwork(tf.keras.Model):
    def __init__(self, input_dim, middle_dim, emb_dim, seq_len, num_nodes, scene_dim, edge_importance):
        super(UnPIENetwork, self).__init__()

        self.gcn = UnPIESTGCN(input_dim, middle_dim, emb_dim, seq_len, num_nodes, scene_dim, edge_importance)
        self.temporal_aggregator = UnPIETemporalAggregator(emb_dim)
    
    def __call__(self, x, b, a, training): 
        '''
        Args:
            x: tf.Tensor, shape=(batch_size, seq_len, num_nodes, num_channels)
            b: tf.Tensor, shape=(batch_size, seq_len, num_nodes, 4)
            a: tf.Tensor, shape=(batch_size, seq_len, num_nodes, num_nodes)
            y: tf.Tensor, shape=(batch_size)
        '''
        # Aggregate spatial features using a GNN
        ped_feat = self.gcn(x, b, a, training) # ped_feat shape: (batch_size, seq_len, emb_dim)

        # Aggregate temporal features
        ped_feat = self.temporal_aggregator(ped_feat) # ped_feat shape: (batch_size, emb_dim)
        return ped_feat