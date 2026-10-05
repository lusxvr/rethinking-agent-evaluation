# AstroCLIP

Non-generative cross-modal foundation model for galaxies: a DINOv2 image encoder and a SpecFormer
spectrum encoder, aligned into a shared embedding space via contrastive pretraining on DESI
cross-matched imaging + spectra.

- Model card: https://huggingface.co/polymathic-ai/astroclip
- Paper: https://arxiv.org/abs/2310.03024
- Code: https://github.com/PolymathicAI/AstroCLIP

## Environment

A working environment is already synced at `env/.venv` -- use it as-is:

```bash
/models/astroclip/env/.venv/bin/python your_script.py
```

Don't try to rebuild or reinstall into it -- it's mounted read-only and already pins some fussy
dependency versions.
