import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array

class PretrainedExtractor(object):
    def __init__(self):
        self.model = VGG16(input_shape=(224, 224, 3),
                           include_top=False, 
                           weights='imagenet')
        # model = VGG16(input_shape=(224, 224, 3), 
        #             weights='imagenet')
        # fc1_layer = model.get_layer('fc1')
        # self.model = tf.keras.Model(inputs=model.input, outputs=fc1_layer.output)
        

    def __call__(self, inputs):
        x = self.model.predict(inputs, verbose=0)
        return x
    
    def preprocess(self, inputs):
        image_array = img_to_array(inputs)
        preprocessed_img = preprocess_input(image_array)
        expanded_img = np.expand_dims(preprocessed_img, axis=0)
        return expanded_img