import tensorflow as tf

from model.feature_extractor import FeatureExtractor
from model.temporal_aggregator import TemporalAggregator


class UnPIENetwork(object):
    def __init__(self, emb_size, dropout_rate1, dropout_rate2):
        self.feature_extractor = FeatureExtractor(emb_size, dropout_rate1, dropout_rate2)
        self.temporal_aggregator = TemporalAggregator(emb_size)
    
    def __call__(self, ped_feat, ped_bbox): 
        '''
        Args:
            ped_feat: [batch_size, num_frames, emb_size]
            ped_bbox: [batch_size, num_frames, 4]
            objs_feat: [batch_size, num_frames, num_objs, emb_size]
            objs_bbox: [batch_size, num_frames, num_objs, 4]
            other_peds_feat: [batch_size, num_frames, num_other_peds, emb_size]
            other_peds_bbox: [batch_size, num_frames, num_other_peds, 4]
        '''
        # Compute features
        new_ped_feat = self.feature_extractor(ped_feat, ped_bbox)

        # Extract features for each object and other pedestrian: TODO
        # Aggregate spatial features using a GNN: TODO

        # Aggregate temporal features
        new_ped_feat = self.temporal_aggregator(new_ped_feat)
        return new_ped_feat