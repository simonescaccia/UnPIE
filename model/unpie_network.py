import tensorflow as tf

from model.feature_extractor import FeatureExtractor
from model.temporal_aggregator import TemporalAggregator


class UnPIENetwork(object):
    def __init__(self, middle_size, emb_size, dropout_rate1, dropout_rate2):
        self.feature_extractor = FeatureExtractor(middle_size, emb_size, dropout_rate1, dropout_rate2)
        self.temporal_aggregator = TemporalAggregator(emb_size)
    
    def __call__(self, ped_feat, ped_bbox, objs_feat, objs_bbox, other_peds_feat, other_peds_bbox): 
        '''
        Args:
            x: tf.Tensor, shape (batch_size, num_frames, crop_size, crop_size, num_channels)
            b: tf.Tensor, shape (batch_size, num_frames, 4)
        '''
        ped_feat = self.feature_extractor(ped_feat, ped_bbox)
        ped_feat = self.temporal_aggregator(ped_feat)
        return ped_feat