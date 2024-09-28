from model.feature_extractor import FeatureExtractor
from model.temporal_aggregator import TemporalAggregator


class UnPIEFramework(object):
    def __init__(self, emb_size):
        self.feature_extractor = FeatureExtractor(emb_size)
        self.temporal_aggregator = TemporalAggregator()
    
    def __call__(self, x):
        x = self.feature_extractor(x)
        x = self.temporal_aggregator(x)
        return x