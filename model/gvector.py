import torch
import pdb
import torch.nn as nn
from .resnet import ResNet, BasicBlock, Bottleneck


class Gvector(nn.Module):
    def __init__(self, channels, num_blocks, embd_dim, drop):
        super(Gvector, self).__init__()
        self.resnet = ResNet(channels, BasicBlock, num_blocks)
        self.fc = nn.Linear(channels * 8 * 2, embd_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.resnet(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x_mean = x.mean(dim=2)
        x_std = x.std(dim=2)
        x = torch.cat((x_mean, x_std), dim=1)
        x = self.fc(x)
        x = self.dropout(x)
        return x
