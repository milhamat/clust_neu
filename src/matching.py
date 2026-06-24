import torch
import torch.nn as nn

from backbone import ResNet18Embedding
from config import EMBED_DIM


class MatchingNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = ResNet18Embedding()

        # Full Context Embedding (Support Set)
        self.fce = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=EMBED_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # BiLSTM output = EMBED_DIM * 2
        self.projection = nn.Linear(
            EMBED_DIM * 2,
            EMBED_DIM
        )

        # Query Context Encoder
        self.query_lstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=EMBED_DIM,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):

        emb = self.encoder(x)

        return emb

    # =====================================
    # SUPPORT FCE
    # =====================================

    def contextualize(
        self,
        support_embs
    ):

        # support_embs
        # [N, EMBED_DIM]

        out, _ = self.fce(
            support_embs.unsqueeze(0)
        )

        # [1, N, EMBED_DIM*2]
        out = out.squeeze(0)

        out = self.projection(
            out
        )

        # residual connection
        out = support_embs + out

        return out

    # =====================================
    # QUERY FCE
    # =====================================

    def contextualize_query(
        self,
        query_emb
    ):

        # query_emb
        # [1, EMBED_DIM]

        out, _ = self.query_lstm(
            query_emb.unsqueeze(0)
        )

        # [1,1,EMBED_DIM]
        out = out.squeeze(0)

        # residual connection
        out = query_emb + out

        return out