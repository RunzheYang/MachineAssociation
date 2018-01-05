import os
import os.path
import torch.utils.data as data
import torch

DATA_FILE_NAME = "mnist_apprentice.data"


class APPRENTICE(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data, self.labels = torch.load(
            os.path.join(self.root, DATA_FILE_NAME)
        )

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)
