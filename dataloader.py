import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize

from TRAINING_CONFIG import *


class UnderwaterDataset(Dataset):
    def __init__(self, image_A_dir, image_B_dir, transform=None):
        self.image_A_dir = image_A_dir
        self.image_B_dir = image_B_dir
        self.transform = transform

        self.image_A_list = sorted(os.listdir(self.image_A_dir))
        self.image_B_list = sorted(os.listdir(self.image_B_dir))

    def __len__(self):
        return len(self.image_A_list)

    def __getitem__(self, idx):
        image_A_path = os.path.join(self.image_A_dir, self.image_A_list[idx])
        image_B_path = os.path.join(self.image_B_dir, self.image_B_list[idx])

        image_A = Image.open(image_A_path).convert("RGB")
        image_B = Image.open(image_B_path).convert("RGB")

        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return {'input': image_A, 'label': image_B}


def data_prep():
    transform = Compose([Resize((train_img_size, train_img_size)), ToTensor()])
    train_dataset = UnderwaterDataset(raw_image_path, clear_image_path, transform=transform)
    val_dataset = UnderwaterDataset(VAL_DATA_PATH, VAL_DATA_PATH, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader
