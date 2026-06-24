import torch.nn as nn

from torchvision import models

from config import EMBED_DIM


class ResNet18Embedding(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.embedding = nn.Sequential(

            nn.Linear(512, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, EMBED_DIM)

        )

    def forward(self, x):

        x = self.backbone(x)

        x = self.embedding(x)

        return x