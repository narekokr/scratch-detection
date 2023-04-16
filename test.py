import sys

import numpy as np
import torch
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt, cm
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader

from dataset import ScratchDataset
from model import UNet

model = UNet()
model = nn.DataParallel(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('checkpoints/model(1).pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
dataset = ScratchDataset(damaged_dir='test_data/imgs', mask_dir='test_data/masks', image_size=(256, 256))
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

with torch.no_grad():
    for data in dataloader:
        inputs, masks = data
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        img = masks[0].permute(1, 2, 0)
        img = img.numpy()
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img)
        mask = outputs[0].permute(1, 2, 0).numpy() > 0.5
        a = ndimage.binary_dilation(np.squeeze(mask, axis=2)).astype(mask.dtype)
        ax2.imshow(a)
        plt.show()
        plt.imsave('mask.png', a, cmap=cm.gray)
