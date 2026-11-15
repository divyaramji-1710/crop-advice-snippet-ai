"""Causal transformer components implemented from scratch."""
import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size)),
        )

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weights = q @ k.transpose(-2, -1)
        weights = weights * k.shape[-1] ** -0.5
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf"),
        )
        weights = F.softmax(weights, dim=-1)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()
        head_size = n_embd // n_heads
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size)
            for _ in range(n_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        combined = torch.cat(
            [head(x) for head in self.heads],
            dim=-1,
        )
        return self.proj(combined)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()

        if n_embd < 1 or n_heads < 1 or block_size < 1:
            raise ValueError("Model dimensions must be positive")
        if n_embd % n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads")

        self.sa = MultiHeadAttention(
            n_embd, n_heads, block_size
        )
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=32,
        n_embd=64,
        n_heads=4,
        n_layers=3,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_heads, block_size)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T < 1 or T > self.block_size:
            raise ValueError(
                "Sequence length is outside the context window"
            )

        positions = torch.arange(T, device=idx.device)
        x = (
            self.token_embedding(idx)
            + self.position_embedding(positions)
        )
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(B * T, -1),
                targets.reshape(B * T),
            )

        return logits, loss