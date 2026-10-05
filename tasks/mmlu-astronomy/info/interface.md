## Model interface

Load it with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

weights_dir = "/models/astrosage/weights"
tokenizer = AutoTokenizer.from_pretrained(weights_dir)
model = AutoModelForCausalLM.from_pretrained(weights_dir, dtype=torch.bfloat16, device_map="cuda")
model.eval()
```

It is a plain causal LM: tokenize a prompt, call `model.generate(**inputs,
pad_token_id=tokenizer.eos_token_id)`, and decode only the newly generated tokens (i.e. skip the
input prompt's own token span).
