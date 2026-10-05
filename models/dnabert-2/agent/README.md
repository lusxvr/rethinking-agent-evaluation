DNABERT-2 (117M) is a BERT-style encoder for genomic DNA sequences, pretrained with masked
language modeling on a multi-species genome corpus. It uses byte-pair-encoding tokenization over
raw nucleotide sequences rather than fixed-length k-mers.

- Model card: https://huggingface.co/zhihan1996/DNABERT-2-117M
- Paper: https://arxiv.org/abs/2306.15006
- Code: https://github.com/MAGICS-LAB/DNABERT_2

## Environment

A working environment is already synced at `env/.venv` -- use it as-is:

```bash
/models/dnabert-2/env/.venv/bin/python your_script.py
```

Don't try to rebuild or reinstall into it -- it's mounted read-only and already pins some fussy
dependency versions.
