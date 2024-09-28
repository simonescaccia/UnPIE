from typing import Any
import tensorflow as tf

class PretrainedExtractor(object):
    def __init__(self):
        self.model = tf.keras.applications.VGG16(input_shape=(224, 224, 3),
                                                include_top=False,
                                                weights='imagenet')

    def __call__(self, inputs):
        x = self.model.predict(inputs, verbose=0) # shape: (batch_size, 7, 7, 512)
        return x