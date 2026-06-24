import torch
import torch.nn as nn

from backbone import ResNet18Embedding


class MatchingNetwork(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        hidden_dim=128
    ):

        super().__init__()

        self.encoder = ResNet18Embedding()

        self.fce = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.projection = nn.Linear(
            hidden_dim * 2,
            embed_dim
        )

    def forward(self, x):

        emb = self.encoder(x)

        return emb

    def contextualize(
        self,
        support_embs
    ):

        out, _ = self.fce(
            support_embs.unsqueeze(0)
        )

        out = self.projection(
            out.squeeze(0)
        )

        return out

# import torch
# import torch.nn as nn

# from backbone import ResNet18Embedding


# class MatchingNetwork(nn.Module):

#     def __init__(self):

#         super().__init__()

#         self.encoder = ResNet18Embedding()

#     def forward(self, x):

#         return self.encoder(x)
    
    