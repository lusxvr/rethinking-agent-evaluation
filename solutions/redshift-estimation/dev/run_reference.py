"""REFERENCE anchor: embed each query image and predict redshift with AstroCLIP's zero_shot() k-NN
(k=64, standardized, Euclidean, distance-weighted) over the reference catalog.

Usage:
    cd models/astroclip/dev_env
    uv run --project . python ../../../solutions/redshift-estimation/dev/run_reference.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from astroclip.data.datamodule import AstroClipCollator
from astroclip.models import AstroClipModel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from eval.evaluate import evaluate  # noqa: E402

WEIGHTS_PATH = REPO_ROOT / "models" / "astroclip" / "agent" / "weights" / "astroclip.ckpt"
DATA_DIR = REPO_ROOT / "tasks" / "redshift-estimation" / "data"
RESULTS_DIR = REPO_ROOT / "solutions" / "redshift-estimation" / "dev" / "results"

K_NEIGHBORS = 64

# AstroCLIP's training preprocessing; must match prepare_data.py's reference embeddings.
_COLLATOR = AstroClipCollator()


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading AstroCLIP from {WEIGHTS_PATH} onto {device}...")
    model = AstroClipModel.load_from_checkpoint(
        checkpoint_path=str(WEIGHTS_PATH), map_location=device, weights_only=False
    )
    model.eval()

    catalog = np.load(DATA_DIR / "reference_catalog.npz")
    reference_embeddings = catalog["embeddings"]  # (N_REFERENCE, 1024)
    reference_redshift = catalog["redshift"]  # (N_REFERENCE,)

    query_paths = sorted((DATA_DIR / "query_images").glob("*.npy"))
    query_images = [np.load(p) for p in query_paths]  # each (152, 152, 3), raw nanomaggie flux
    targetids = [p.stem for p in query_paths]

    samples = [{"image": torch.from_numpy(img)} for img in query_images]
    tensor = _COLLATOR(samples)["image"].to(device)
    with torch.no_grad():
        query_embeddings = model(tensor, input_type="image").cpu().numpy()  # (n_query, 1024)

    # Standardize per-dimension (fit on the reference pool only) -- required before Euclidean
    # distance is meaningful across 1024 dimensions of very different scale/variance.
    scaler = StandardScaler().fit(reference_embeddings)
    reference_scaled = scaler.transform(reference_embeddings)
    query_scaled = scaler.transform(query_embeddings)

    knn = KNeighborsRegressor(n_neighbors=K_NEIGHBORS, weights="distance")
    knn.fit(reference_scaled, reference_redshift)
    predictions = knn.predict(query_scaled)  # (n_query,)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "reference_prediction.csv"
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["targetid", "redshift"])
        for targetid, redshift in zip(targetids, predictions):
            writer.writerow([targetid, redshift])
    print(f"Wrote {output_path}")

    result = evaluate("redshift-estimation", output_path)
    if result["valid"]:
        print(f"reference_R2 = {result['score']:.4f} ({result['n']} rows)")
    else:
        print("INVALID:", result["errors"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
