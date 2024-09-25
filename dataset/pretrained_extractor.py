from typing import Any
import tensorflow as tf

class PretraineExtractor(object):
    def __init__(self):
        self.final_size = 128
        self.model = tf.keras.applications.VGG16(input_shape=(224, 224, 3),
                                                include_top=False,
                                                weights='imagenet')
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(self.final_size)

    def __call__(self, inputs):
        x = self.model.predict(inputs, verbose=0) # shape: (batch_size, 7, 7, 512)
        x = self.global_pool(x) # shape: (batch_size, 512)
        x = self.fc(x) # shape: (batch_size, final_size)
        return x