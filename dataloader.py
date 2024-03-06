import os

import torch
from torch.utils.data import Dataset

PREFIX_DATA = "dataimage"
PREFIX_LABEL = "error"


class RandomKappaDataset(Dataset):
    def __init__(self, folder_path, transform=None, target_transform=None) -> None:
        self.transform = transform
        self.target_transform = target_transform
        self.folder_path = folder_path

    def __len__(self):
        return len([name for name in os.listdir(self.folder_path)
                    if os.path.isfile(os.path.join(self.folder_path, name)) and name.startswith(PREFIX_DATA)])

    def __getitem__(self, index):
        image = torch.load(f"{self.folder_path}/{PREFIX_DATA}{str(index)}.pt")
        label = torch.load(f"{self.folder_path}/{PREFIX_LABEL}{str(index)}.pt")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        image[2, :, :] = image[2, :, :] ** 2
        return image, label
