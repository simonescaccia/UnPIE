import torch.utils.data as data


class PIEDataset(data.Dataset):
    def __init__(self, features):
        self.ped_feats = features['ped_feats']
        self.ped_bboxes = features['ped_bboxes']
        self.ped_labels = features['intention_binary']
        assert len(self.ped_feats) == len(self.ped_bboxes) == len(self.ped_labels)
    def __getitem__(self, index):
        return self.ped_feats[index], self.ped_bboxes[index], self.ped_labels[index], index

    def __len__(self):
        return len(self.ped_feats)