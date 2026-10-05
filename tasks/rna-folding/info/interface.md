## Model interface

RiNALMo is loaded as two separate PyTorch modules: the language model backbone (`RiNALMo`) and a
secondary-structure prediction head (`SecStructPredictionHead`) that sits on top of it. Both come
from the `rinalmo` package (already installed in your environment):

```python
from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet
from rinalmo.model.downstream import SecStructPredictionHead
from rinalmo.model.model import RiNALMo

device = "cuda" if torch.cuda.is_available() else "cpu"  # this checkpoint is small enough that
                                                           # CPU-only inference silently "works" but
                                                           # is far slower -- always prefer the GPU

config = model_config("giga")
lm = RiNALMo(config)
pred_head = SecStructPredictionHead(config.model.transformer.embed_dim, num_blocks=2)

state_dict = torch.load("/models/rinalmo/weights/rinalmo_giga_ss_bprna_ft.pt", map_location=device)
threshold = state_dict.pop("threshold")  # a float, tuned when this checkpoint was made
lm_state = {k[len("lm."):]: v for k, v in state_dict.items() if k.startswith("lm.")}
head_state = {k[len("pred_head."):]: v for k, v in state_dict.items() if k.startswith("pred_head.")}
lm.load_state_dict(lm_state, strict=False)  # strict=True raises: the checkpoint never carries
                                             # rotary_emb.inv_freq (a fixed, non-learned buffer,
                                             # not something you need to supply)
pred_head.load_state_dict(head_state)
lm = lm.to(device).eval()
pred_head = pred_head.to(device).eval()

alphabet = Alphabet(**config.alphabet)
tokens = torch.tensor([alphabet.encode(sequence)], device=device)  # adds start/end tokens automatically
representation = lm(tokens)["representation"]         # (1, len(sequence)+2, 1280)
logits = pred_head(representation[:, 1:-1, :]).squeeze(-1)  # strip start/end -> (1, L, L)
probs = torch.sigmoid(logits)                          # symmetric base-pair probability matrix
```

`probs[0, i, j]` is the model's estimated probability that position `i` and position `j` are
base-paired. This is a raw probability matrix, not yet a valid secondary structure -- turning it
into one (respecting biological constraints, resolving conflicts so each base has at most one
partner, and choosing the actual pairing threshold) is additional work this interface doesn't do
for you.

`threshold`, popped out of the checkpoint's own state dict above, is a value the checkpoint's own
training tuned specifically for this task -- not a fixed default like 0.5.
