## Model interface

Load the checkpoint:

```python
from astroclip.models import AstroClipModel

model = AstroClipModel.load_from_checkpoint(
    checkpoint_path="/models/astroclip/weights/astroclip.ckpt",
    map_location=device,
    weights_only=False,
)
model.eval()
```

Call the model on a batch of preprocessed image tensors to get embeddings:
`model(tensor, input_type="image")` returns a `(batch, 1024)` embedding. Always embed in batches
of 2 or more -- a batch of exactly 1 crashes on an upstream bug in the model's cross-attention
head.
