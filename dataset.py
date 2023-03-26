import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ScratchDataset(Dataset):
    def __init__(self, damaged_dir, mask_dir, image_size=(256, 256)):
        self.damaged_dir = damaged_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        self.transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.damaged_files = os.listdir(self.damaged_dir)
        self.mask_files = os.listdir(self.mask_dir)
        self.masks_num = len(self.mask_files)

        self.toTensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.damaged_files)

    def __getitem__(self, index):
        damaged_path = os.path.join(self.damaged_dir, self.damaged_files[index])
        mask_index = random.randint(0, self.masks_num - 1)
        mask_path = os.path.join(self.mask_dir, self.mask_files[mask_index])

        damaged_img = Image.open(damaged_path).convert('L')
        mask_img = Image.open(mask_path).convert('1')

        if self.transform:
            damaged_img = self.transform(damaged_img)
            mask_img = self.mask_transform(mask_img)
        return damaged_img, mask_img
