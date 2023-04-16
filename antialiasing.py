import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Downsample(nn.Module):
    def __init__(self, pad_type="reflect", filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels
        a = np.array([1.0, 2.0, 1.0])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = nn.ReflectionPad2d(self.pad_sizes)

    def forward(self, inp):
        return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])