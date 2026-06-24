import torch
import torch.nn as nn

from backbone import ResNet18Embedding


class ProtoNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = ResNet18Embedding()

    def forward(self, x):

        return self.encoder(x)