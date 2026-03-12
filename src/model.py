import torch
import torch.nn as nn


class GPTEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.token_emb = nn.Embedding(
            config.vocab_size,
            config.emb_dim
        )

        self.pos_emb = nn.Embedding(
            config.context_length,
            config.emb_dim
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):

        batch_size, seq_len = x.shape

        token_embeddings = self.token_emb(x)

        positions = torch.arange(seq_len, device=x.device)

        position_embeddings = self.pos_emb(positions)

        x = token_embeddings + position_embeddings

        return self.dropout(x)