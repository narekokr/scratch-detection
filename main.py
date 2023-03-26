import argparse
import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataset import ScratchDataset
from model import UNet

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define the model
model = UNet()
model = nn.DataParallel(model)
model.to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

criterion = DiceLoss()

# Define the dataset and dataloader
dataset = ScratchDataset(damaged_dir='train_data/imgs', mask_dir='train_data/masks', image_size=(256, 256))
train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Train the model
start = time.time()

for epoch in range(epochs):
    for i, data in enumerate(dataloader):
        inputs, masks = data
        inputs, masks = inputs.to(device), masks.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        # with torch.no_grad():
        #     outputs = torch.sigmoid(outputs)
        #     img = inputs[0].permute(1, 2, 0)
        #     img = img.numpy()
        #     f, (ax1, ax2) = plt.subplots(1, 2)
        #     ax1.imshow(img)
        #     ax2.imshow(masks[0].permute(1, 2, 0).numpy())
        #     plt.show()

        taken = datetime.timedelta(seconds=(time.time() - start))
        formatted_time = str(taken).split('.')[0]
        print('[%d, %5d] loss: %.3f, time taken:' % (epoch + 1, i + 1, loss.item()), formatted_time)

torch.save(model.state_dict(), 'model.pt')
