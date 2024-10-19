import torch.utils.data as data


class PIEDataset(data.Dataset):
    def __init__(self, features):
        self.ped_feats = features['ped_feats']
        self.ped_bboxes = features['ped_bboxes']
        self.ped_labels = features['intention_binary']
        self.obj_feats = features['obj_feats']
        self.obj_bboxes = features['obj_bboxes']
        self.other_ped_feats = features['other_ped_feats']
        self.other_ped_bboxes = features['other_ped_bboxes']
        assert len(self.ped_feats) == len(self.ped_bboxes) == len(self.ped_labels) \
            == len(self.obj_feats) == len(self.obj_bboxes) == len(self.other_ped_feats) \
            == len(self.other_ped_bboxes)

    def __getitem__(self, index):
        return self.ped_feats[index], self.ped_bboxes[index], self.ped_labels[index], \
               self.obj_feats[index], self.obj_bboxes[index], self.other_ped_feats[index], self.other_ped_bboxes[index], \
               index

    def __len__(self):
        return len(self.ped_feats)