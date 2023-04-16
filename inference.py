import numpy as np
import torch
from matplotlib import pyplot as plt, cm
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader

from dataset import ScratchDataset, InferenceDataset
from model import UNet

model = UNet()
model = nn.DataParallel(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('checkpoints/model (2).pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
dataset = InferenceDataset(damaged_dir='imgs', image_size=(512, 512))
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

with torch.no_grad():
    for i, data in enumerate(dataloader):
        inputs, names = data
        print(inputs.size())
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        for image, name in zip(outputs, names):
            mask = image.permute(1, 2, 0).numpy() > 0.5
            dilated = ndimage.binary_dilation(np.squeeze(mask, axis=2), iterations=2).astype(mask.dtype)
            plt.imsave(f'results/{name}', dilated, cmap = plt.cm.gray)

        # mask = outputs[0].permute(1, 2, 0).numpy() > 0.5
        # a = ndimage.binary_dilation(np.squeeze(mask, axis=2)).astype(mask.dtype)
        # f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        #
        # ax1.set_title('original mask')
