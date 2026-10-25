import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.LocalResponseNorm as LRN

from deepul_helper.batch_norm import BatchNorm2d

class SpatialContext(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SpatialContext, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x