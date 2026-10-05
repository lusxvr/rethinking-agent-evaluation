"""Stub -- see models/rinalmo/README.md's flash-attn gotcha and flash_attn/__init__.py."""

from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "flash_attn is stubbed out in this environment -- RiNALMo must be run with "
            "use_flash_attn=False, which never instantiates or calls this class."
        )
