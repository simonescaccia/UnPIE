import tensorflow as tf

class TemporalAggregator(object):
    def __init__(self):
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
    
    def __call__(self, inputs): # inputs shape: (batch_size, seq_length, final_size)
        x = self.global_pool(inputs) # shape: (batch_size, final_size)
        return x