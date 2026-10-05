## Model interface

DNABERT-2 is a BERT-style encoder over raw DNA sequences (A/C/G/T strings), using its own
byte-pair-encoding tokenizer rather than fixed-length k-mers. It's loaded via `trust_remote_code`,
which requires an explicit config object with `transformers>4.28`:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig

tokenizer = AutoTokenizer.from_pretrained("/models/dnabert-2/weights", trust_remote_code=True)
config = BertConfig.from_pretrained("/models/dnabert-2/weights", num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(
    "/models/dnabert-2/weights", trust_remote_code=True, config=config
)
```

This attaches a fresh, untrained classification head (the released checkpoint is a pretrained
encoder only) -- you should expect a "newly initialized weights" warning naming
`bert.pooler`/`classifier`, and to need to train that head (and, per the paper, the full encoder
too) before it produces a meaningful classification.

Sequences longer than 512 tokens are truncated (`max_position_embeddings=512`). The model's
`SequenceClassifierOutput` always populates `hidden_states` (never `None`), so code that expects a
bare logits tensor from a training loop's prediction-gathering step should account for a
`(logits, hidden_states)` tuple instead.
