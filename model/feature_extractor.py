import tensorflow as tf

class FeatureExtractor(object):
    def __init__(self, middle_size, emb_size, dropout_rate1, dropout_rate2):
        self.fc1 = tf.keras.layers.Dense(middle_size, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate1)
        self.fc2 = tf.keras.layers.Dense(emb_size, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate2)

    def __call__(self, x, b):
        '''
        Args:
            x: tf.Tensor, shape (batch_size, num_frames, crop_size, crop_size, num_channels)
            b: tf.Tensor, shape (batch_size, num_frames, 4)
        '''
        assert x.get_shape()[0] == b.get_shape()[0]
        assert x.get_shape()[1] == b.get_shape()[1]
        assert b.get_shape()[2] == 4

        x_shape = x.get_shape().as_list()
        x = tf.reshape(x, [x_shape[0], x_shape[1], -1]) # x shape: (batch_size, num_frames, crop_size*crop_size*num_channels)
        x = self.fc1(x) # x shape: (batch_size, num_frames, num_channels)
        x = self.dropout1(x)
        x = tf.concat([x, b], axis=-1) # x shape: (batch_size, num_frames, num_channels+4)
        x = self.fc2(x) # x shape: (batch_size, num_frames, emb_size)
        x = self.dropout2(x)
        return x