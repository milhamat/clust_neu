import torch.nn as nn

from backbone import ResNet18Embedding


class SiameseNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = ResNet18Embedding()

    def forward(self, x1, x2):

        z1 = self.encoder(x1)

        z2 = self.encoder(x2)

        return z1, z2


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):

        super().__init__()

        self.margin = margin

    def forward(self, output1, output2, label):

        distance = nn.functional.pairwise_distance(
            output1,
            output2
        )

        loss = (

            (1 - label) *
            distance.pow(2)

            +

            label *
            (
                self.margin - distance
            ).clamp(min=0).pow(2)

        )

        return loss.mean()