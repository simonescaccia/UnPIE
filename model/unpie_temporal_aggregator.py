import tensorflow as tf

class UnPIETemporalAggregator(object):
    def __init__(self, emb_dim):
        self.lstm = tf.keras.layers.LSTM(emb_dim)
    
    def __call__(self, x): # inputs shape: (batch_size, num_frames, emb_dim)
        x = self.lstm(x) # shape: (batch_size, emb_dim)
        return x