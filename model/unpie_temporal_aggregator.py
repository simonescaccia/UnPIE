import tensorflow as tf

class UnPIETemporalAggregator(tf.keras.Model):
    def __init__(self, **params):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(
            units=params['emb_dim'],
            dropout=params['drop_lstm']
        )
    
    def call(self, x): # inputs shape: (batch_size, seq_len, emb_dim)
        x = self.lstm(x) # shape: (batch_size, emb_dim)
        return x