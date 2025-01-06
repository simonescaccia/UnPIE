import tensorflow as tf

from model.attention import AttentionLayer
from model.unpie_st_gcn import UnPIESTGCN
from model.unpie_temporal_aggregator import UnPIETemporalAggregator

class UnPIENetwork(tf.keras.Model):
    def __init__(self, **params):
        super().__init__()

        self.gcn = UnPIESTGCN(**params)
        self.attention = AttentionLayer(input_dim=params['emb_dim'])
        self.temporal_aggregator = UnPIETemporalAggregator(**params)

        self.fcn = tf.keras.layers.Dense(1, activation='sigmoid')

        self.task = params['task']

    def call(self, inputs, training):
        x, b, c, a = inputs
        '''
        Args:
            x: tf.Tensor, shape=(batch_size, seq_len, num_nodes, num_channels)
            b: tf.Tensor, shape=(batch_size, seq_len, num_nodes, 4)
            a: tf.Tensor, shape=(batch_size, seq_len, num_nodes, num_nodes)
            y: tf.Tensor, shape=(batch_size)
        '''
        # Aggregate spatial features using a GNN
        ped_feat = self.gcn(x, b, c, a, training) # ped_feat shape: (batch_size, seq_len, emb_dim)
        # Aggregate temporal features
        ped_feat = self.temporal_aggregator(ped_feat) # ped_feat shape: (batch_size, emb_dim)

        if self.task == 'SUP':
            ped_feat = self.fcn(ped_feat) # ped_feat shape: (batch_size, 1)
        
        return ped_feat