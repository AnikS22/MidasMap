"""
Minimal ResNet-50 definition without torchvision dependency.
Used only for feature extraction from saved checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch

        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            x = self.downsample(x)
        out = out + x
        out = self.relu(out)
        return out


class BottleneckBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        mid_ch = out_ch // 4
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            x = self.downsample(x)
        out = out + x
        out = self.relu(out)
        return out


class MinimalResNet50(nn.Module):
    """Minimal ResNet-50 for feature extraction."""

    def __init__(self):
        super().__init__()

        # Stem
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Layers (ResNet-50: 3, 4, 6, 3 blocks)
        self.layer1 = self._make_layer(64, 256, 3, stride=1)      # 256ch
        self.layer2 = self._make_layer(256, 512, 4, stride=2)     # 512ch
        self.layer3 = self._make_layer(512, 1024, 6, stride=2)    # 1024ch
        self.layer4 = self._make_layer(1024, 2048, 3, stride=2)   # 2048ch

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = []
        layers.append(BottleneckBlock(in_ch, out_ch, stride=stride))
        for _ in range(num_blocks - 1):
            layers.append(BottleneckBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        p2 = self.layer1(x)
        p3 = self.layer2(p2)
        p4 = self.layer3(p3)
        p5 = self.layer4(p4)
        return p2, p3, p4, p5
