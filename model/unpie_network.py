import tensorflow as tf

from model.feature_extractor import FeatureExtractor
from model.temporal_aggregator import TemporalAggregator


class UnPIENetwork(object):
    def __init__(self, emb_size, dropout_rate1, dropout_rate2):
        self.feature_extractor = FeatureExtractor(emb_size, dropout_rate1, dropout_rate2)
        self.temporal_aggregator = TemporalAggregator(emb_size)
    
    def __call__(self, ped_feat, ped_bbox, objs_feat, objs_bbox, other_peds_feat, other_peds_bbox): 
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

        new_objs_feat = []
        for obj_idx in range(objs_feat.shape[2]):
            obj_feat = objs_feat[:, :, obj_idx, :]
            obj_bbox = objs_bbox[:, :, obj_idx, :]
            obj_feat = self.feature_extractor(obj_feat, obj_bbox)
            new_objs_feat.append(obj_feat)
        new_objs_feat = tf.stack(new_objs_feat, axis=2)
        
        new_other_peds_feat = []
        for other_ped_idx in range(other_peds_feat.shape[2]):
            other_ped_feat = other_peds_feat[:, :, other_ped_idx, :]
            other_ped_bbox = other_peds_bbox[:, :, other_ped_idx, :]
            other_ped_feat = self.feature_extractor(other_ped_feat, other_ped_bbox)
            new_other_peds_feat.append(other_ped_feat)
        new_other_peds_feat = tf.stack(new_other_peds_feat, axis=2)

        # Aggregate spatial features using a GNN
        

        # Aggregate temporal features
        new_ped_feat = self.temporal_aggregator(new_ped_feat)
        return new_ped_feat