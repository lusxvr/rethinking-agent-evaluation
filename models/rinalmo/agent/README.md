RiNALMo (Giga, 650M) is an RNA language model, pretrained with masked language modeling on a
large corpus of non-coding RNA sequences. This checkpoint is further fine-tuned for secondary
structure prediction on the bpRNA dataset -- it outputs a base-pair probability matrix, not just a
raw embedding.

- Model card / weights: https://zenodo.org/records/15043668
- Paper: https://arxiv.org/abs/2403.00043
- Code: https://github.com/lbcb-sci/RiNALMo

## Environment

A working environment is already synced at `env/.venv` -- use it as-is:

```bash
/models/rinalmo/env/.venv/bin/python your_script.py
```

Don't try to rebuild or reinstall into it -- it's mounted read-only and already pins some fussy
dependency versions.
