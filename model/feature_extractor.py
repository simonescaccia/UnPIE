import tensorflow as tf

class FeatureExtractor(object):
    def __init__(self, emb_size):
        self.final_size = emb_size
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(self.final_size)

    def __call__(self, x): # x shape: (batch_size, num_frames, crop_size, crop_size, num_channels)
        x_shape = x.get_shape().as_list()
        x = tf.reshape(x, [-1, x_shape[2], x_shape[3], x_shape[4]]) # x shape: (batch_size*num_frames, crop_size, crop_size, num_channels)
        x = self.global_pool(x) # x shape: (batch_size*num_frames, num_channels)
        x = tf.reshape(x, [-1, x_shape[1], x_shape[4]]) # x shape: (batch_size, num_frames, num_channels)
        x = self.fc(x) # x shape: (batch_size, num_frames, emb_size)
        return x