from spektral.layers import GCSConv, GlobalAvgPool
import tensorflow as tf

class GCN(object):
    def __init__(self, middle_dim ,emb_dim):
        self.conv1 = GCSConv(middle_dim, activation="relu")
        self.conv2 = GCSConv(middle_dim, activation="relu")
        self.conv3 = GCSConv(middle_dim, activation="relu")
        self.global_pool = GlobalAvgPool()
        self.dense = tf.keras.layers.Dense(emb_dim, activation="softmax")

    def __call__(self, x, a):
        '''
        Args:
            x: tf.Tensor, shape=(batch_size, num_nodes, num_channels + 4)
            a: tf.Tensor, shape=(batch_size, num_nodes, num_nodes)
            i: tf.Tensor, shape=(batch_size,)
        '''
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.conv3([x, a])
        x = self.global_pool(x) # x shape: (batch_size, middle_dim)
        x = self.dense(x) # x shape: (batch_size, emb_dim)
        return x
