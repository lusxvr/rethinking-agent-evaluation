"""Stub for the real `flash-attn` package.

RiNALMo's rinalmo/model/attention.py does `from flash_attn import flash_attn_varlen_qkvpacked_func,
flash_attn_qkvpacked_func` unconditionally at module import time (to define
FlashMultiHeadSelfAttention), even though flash-attn isn't declared as one of RiNALMo's own
pyproject.toml dependencies -- so importing rinalmo.model at all requires *something* importable
as `flash_attn`, regardless of whether the flash path is ever used. This benchmark always runs
RiNALMo with use_flash_attn=False (see models/rinalmo/README.md), which routes through
MultiHeadSelfAttention instead and never calls these two functions -- so a real flash-attn build
(a real CUDA-toolchain-dependent compile) is unnecessary; these stubs only need to exist for the
import statement to succeed.
"""


def flash_attn_qkvpacked_func(*args, **kwargs):
    raise RuntimeError(
        "flash_attn is stubbed out in this environment -- RiNALMo must be run with "
        "use_flash_attn=False, which never calls this function."
    )


def flash_attn_varlen_qkvpacked_func(*args, **kwargs):
    raise RuntimeError(
        "flash_attn is stubbed out in this environment -- RiNALMo must be run with "
        "use_flash_attn=False, which never calls this function."
    )
