import tensorflow as tf

class UnPIETemporalAggregator(tf.keras.Model):
    def __init__(self, emb_dim, **kwargs):
        super(UnPIETemporalAggregator, self).__init__(**kwargs)
        self.lstm = tf.keras.layers.LSTM(emb_dim)
    
    def __call__(self, x): # inputs shape: (batch_size, seq_len, emb_dim)
        print(x.shape)
        x = self.lstm(x) # shape: (batch_size, emb_dim)
        return x