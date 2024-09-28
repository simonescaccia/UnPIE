import tensorflow as tf

class FeatureExtractor(object):
    def __init__(self, emb_size):
        self.final_size = emb_size
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(self.final_size)

    def __call__(self, x): # x shape: (batch_size, 7, 7, 512)
        x = self.global_pool(x) # x shape: (batch_size, 512)
        x = self.fc(x) # x shape: (batch_size, final_size)
        return x