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


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.emb_dim // config.n_heads

        self.q_proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.k_proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.v_proj = nn.Linear(config.emb_dim, config.emb_dim)

        self.out_proj = nn.Linear(config.emb_dim, config.emb_dim)

        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_length, config.context_length))
        )

    def forward(self, x):

        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) / (self.head_dim ** 0.5)

        mask = self.mask[:T, :T]

        att = att.masked_fill(mask == 0, float("-inf"))

        att = torch.softmax(att, dim=-1)

        att = self.dropout(att)

        y = att @ v

        y = y.transpose(1,2).contiguous().view(B, T, C)

        y = self.out_proj(y)

        return y
    


class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.emb_dim, 4 * config.emb_dim),
            nn.GELU(),
            nn.Linear(4 * config.emb_dim, config.emb_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)    
    

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.emb_dim)
        self.attn = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.emb_dim)
        self.ff = FeedForward(config)

    def forward(self, x):

        x = x + self.attn(self.ln1(x))

        x = x + self.ff(self.ln2(x))

        return x 

class GPTModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.embedding = GPTEmbedding(config)

        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_f = nn.LayerNorm(config.emb_dim)

        self.head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, x):

        x = self.embedding(x)

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.head(x)

        return logits       