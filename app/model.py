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