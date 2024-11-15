import tensorflow as tf


from model.unpie_st_gcn import UnPIESTGCN
from model.unpie_temporal_aggregator import UnPIETemporalAggregator

class UnPIENetwork(tf.keras.Model):
    def __init__(self, input_dim, middle_dim, emb_dim):
        super().__init__()

        self.gcn = UnPIESTGCN(input_dim, middle_dim, emb_dim)
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