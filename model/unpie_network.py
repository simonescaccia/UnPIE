import tensorflow as tf

from model.feature_extractor import FeatureExtractor
from model.temporal_aggregator import TemporalAggregator


class UnPIENetwork(object):
    def __init__(self, middle_size, emb_size):
        self.feature_extractor = FeatureExtractor(middle_size, emb_size)
        self.temporal_aggregator = TemporalAggregator()
    
    def __call__(self, x, b): 
        '''
        Args:
            x: tf.Tensor, shape (batch_size, num_frames, crop_size, crop_size, num_channels)
            b: tf.Tensor, shape (batch_size, num_frames, 4)
        '''
        x = self.feature_extractor(x, b)
        x = self.temporal_aggregator(x)
        return x