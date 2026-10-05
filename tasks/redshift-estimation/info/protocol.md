## Protocol

`data/reference_catalog.npz` already holds `embeddings`/`redshift` arrays for 126,907 reference
galaxies, pre-embedded exactly the way your query images need to be.

Preprocess raw images before embedding them:

```python
from astroclip.data.datamodule import AstroClipCollator

collator = AstroClipCollator()
tensor = collator([{"image": torch.from_numpy(img)} for img in query_images])["image"]
```

This applies `ToRGB`, an arcsinh stretch of the raw nanomaggie flux, plus a center-crop -- required
preprocessing, not optional. Skipping it produces near-degenerate embeddings. This must match how
the reference embeddings in `data/reference_catalog.npz` were produced.

Then estimate redshift with a k-NN regressor:

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(reference_embeddings)
reference_scaled = scaler.transform(reference_embeddings)
query_scaled = scaler.transform(query_embeddings)

knn = KNeighborsRegressor(n_neighbors=64, weights="distance")
knn.fit(reference_scaled, reference_redshift)
predictions = knn.predict(query_scaled)
```
