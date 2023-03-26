import argparse
import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ScratchDataset
from loss import DiceLoss
from model import UNet

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--load_optimizer', action=argparse.BooleanOptionalAction)
args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs
checkpoint = args.checkpoint
load_optimizer = args.load_optimizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define the model
model = UNet()
model = nn.DataParallel(model)
model.to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Define the dataset and dataloader
dataset = ScratchDataset(damaged_dir='train_data/imgs', mask_dir='train_data/masks', image_size=(256, 256))
train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

start_epoch = 0

if checkpoint:
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) if load_optimizer else None
    start_epoch = checkpoint['epoch']
    model.eval()
    model.train()
# Train the model
start = time.time()

for epoch in range(start_epoch, epochs):
    for i, data in enumerate(dataloader):
        inputs, masks = data
        inputs, masks = inputs.to(device), masks.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        taken = datetime.timedelta(seconds=(time.time() - start))
        formatted_time = str(taken).split('.')[0]
        print('[%d, %5d] loss: %.3f, time taken:' % (epoch + 1, i + 1, loss.item()), formatted_time)

    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, f'checkpoints/model_{epoch}.pt')

torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, 'checkpoints/model.pt')
