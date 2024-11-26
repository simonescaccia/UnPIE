import numpy as np
import tensorflow as tf

class PretrainedExtractor(object):
    def __init__(self):
        
        # self.model = tf.keras.applications.VGG16(
        #     weights='imagenet', 
        #     pooling='avg',
        #     include_top=False
        # )
        self.model = tf.keras.applications.EfficientNetB3(
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )

        #self.preprocess_func = tf.keras.applications.vgg16.preprocess_input
        self.preprocess_func = tf.keras.applications.efficientnet.preprocess_input

    def __call__(self, inputs):
        x = self.model.predict(inputs, verbose=0)
        return x
    
    def preprocess(self, inputs):
        image_array = tf.keras.utils.img_to_array(inputs)
        preprocessed_img = self.preprocess_func(image_array)
        expanded_img = np.expand_dims(preprocessed_img, axis=0)
        return expanded_img