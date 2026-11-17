"""Lightweight Transformer encoder block operating on YOLO feature maps (B,C,H,W)."""
import torch
import torch.nn as nn


def _sincos_2d(h, w, c, device, dtype):
    """Fixed 2-D sin/cos positional encoding -> (1, h*w, c); works for any resolution."""
    q = c // 4
    omega = 1.0 / (10000 ** (torch.arange(q, device=device, dtype=torch.float32) / max(q, 1)))
    gy, gx = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    oy, ox = gy.flatten()[:, None] * omega[None], gx.flatten()[:, None] * omega[None]
    pe = torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], 1)
    if pe.shape[1] < c:
        pe = torch.nn.functional.pad(pe, (0, c - pe.shape[1]))
    return pe[None].to(dtype)


class TransformerEncoder2D(nn.Module):
    """Pre-norm encoder: x + g1*MHSA(LN(x)) ; x + g2*MLP(LN(x)).

    g1/g2 are learnable per-channel gates initialised to 0, so the block is an exact identity at
    initialisation: pretrained YOLO features are preserved and attention is learned gradually.
    Input/output shape is identical (B, C, H, W) -> compatible with the Detect head.
    """

    def __init__(self, c, heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        assert c % heads == 0, f"channels {c} not divisible by heads {heads}"
        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(c)
        hidden = int(c * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(c, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, c))
        self.g1 = nn.Parameter(torch.zeros(c))
        self.g2 = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)                        # (B, HW, C)
        q = self.norm1(t) + _sincos_2d(h, w, c, x.device, t.dtype)
        t = t + self.g1 * self.attn(q, q, self.norm1(t), need_weights=False)[0]
        t = t + self.g2 * self.mlp(self.norm2(t))
        return t.transpose(1, 2).reshape(b, c, h, w)


def register_modules():
    """Make the block resolvable by name from ultralytics model YAMLs."""
    import ultralytics.nn.tasks as tasks
    tasks.TransformerEncoder2D = TransformerEncoder2D
