"""Stub -- see models/rinalmo/README.md's flash-attn gotcha and flash_attn/__init__.py."""


def unpad_input(*args, **kwargs):
    raise RuntimeError(
        "flash_attn is stubbed out in this environment -- RiNALMo must be run with "
        "use_flash_attn=False, which never calls this function."
    )


def pad_input(*args, **kwargs):
    raise RuntimeError(
        "flash_attn is stubbed out in this environment -- RiNALMo must be run with "
        "use_flash_attn=False, which never calls this function."
    )
