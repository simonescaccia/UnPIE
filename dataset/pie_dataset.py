import torch.utils.data as data


class PIEDataset(data.Dataset):
    def __init__(self, images, bboxes, labels):
        assert len(images) == len(bboxes) == len(labels)
        self.images = images
        self.bboxes = bboxes
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.bboxes[index], self.labels[index], index

    def __len__(self):
        return len(self.images)