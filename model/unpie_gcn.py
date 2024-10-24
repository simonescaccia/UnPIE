from spektral.layers import GCSConv, GlobalAvgPool
from spektral.utils.convolution import normalized_adjacency
import tensorflow as tf

class UnPIEGCN(object):
    transform = normalized_adjacency

    def __init__(self, middle_dim ,emb_dim):
        self.conv1 = GCSConv(channels=middle_dim, activation="relu")
        self.conv2 = GCSConv(channels=middle_dim, activation="relu")
        self.conv3 = GCSConv(channels=middle_dim, activation="relu")
        self.global_pool = GlobalAvgPool()
        self.dense = tf.keras.layers.Dense(emb_dim, activation="softmax")

    def __call__(self, x, a):
        '''
        Args:
            x: tf.Tensor, shape=(batch_size, num_frames, num_nodes, num_channels + 4)
            a: tf.Tensor, shape=(batch_size, num_frames, num_nodes, num_nodes)
            i: tf.Tensor, shape=(batch_size,)
        '''
        x_shape = tf.shape(x)
        a_shape = tf.shape(a)

        x = tf.reshape(x, (x_shape[0] * x_shape[1], x_shape[2], x_shape[3]))
        a = tf.reshape(a, (a_shape[0] * a_shape[1], a_shape[2], a_shape[3]))

        x = self.conv1((x, a)) # x shape: (batch_size * num_frames, num_nodes, middle_dim)
        x = self.conv2((x, a)) # x shape: (batch_size * num_frames, num_nodes, middle_dim)
        x = self.conv3((x, a)) # x shape: (batch_size * num_frames, num_nodes, middle_dim)
        x = self.global_pool(x) # x shape: (batch_size * num_frames, middle_dim)
        x = self.dense(x) # x shape: (batch_size * num_frames, emb_dim)

        x = tf.reshape(x, (x_shape[0], x_shape[1], -1))
        return x
