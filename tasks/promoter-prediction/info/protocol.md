## Protocol

`data/train.csv` (`sequence_id,sequence,label`) is the fine-tuning set. Tokenize with a fixed
length of 20 (the sequences are all 70 nucleotides, and DNABERT-2's byte-pair tokenizer compresses
them well below that):

```python
def tokenize(batch):
    return tokenizer(batch["sequence"], padding="max_length", truncation=True, max_length=20)
```

Fine-tune the full model (encoder + the classification head from the interface snippet above) with
`transformers.Trainer`:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="/agent_run/workspace/run",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    warmup_steps=50,
    fp16=True,
    report_to=[],
)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()
```

Then run the fine-tuned model over `data/query_sequences.csv` (`sequence_id,sequence`, no label),
tokenized the same way, and take `argmax` over the two logits as the predicted label:

```python
import numpy as np

logits = trainer.predict(query_dataset).predictions
if isinstance(logits, (tuple, list)):
    logits = logits[0]  # this model's output always also carries hidden_states -- see interface.md
predictions = np.argmax(logits, axis=-1)
```

Write `sequence_id,label` for all 613 query sequences and call `finish`.
