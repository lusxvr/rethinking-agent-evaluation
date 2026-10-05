"""BACKBONE anchor: one multimodal chat completion per query galaxy (images converted with
AstroCLIP's arcsinh ToRGB stretch). Writes backbone_prediction_{no_thinking,thinking}.csv.

Usage:
    cd models/astroclip/dev_env
    uv run --project . python ../../../solutions/redshift-estimation/dev/run_backbone.py
"""

import argparse
import base64
import csv
import io
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from axes import DEFAULT_MODEL_NAME  # noqa: E402

DATA_DIR = REPO_ROOT / "tasks" / "redshift-estimation" / "data"
RESULTS_DIR = REPO_ROOT / "solutions" / "redshift-estimation" / "dev" / "results"

QUESTION = "Estimate this galaxy's cosmic redshift. Respond with a single decimal number and nothing else."

# Fixed large budget: with thinking, many answers exceed 8192 tokens.
MAX_TOKENS = 60_000

# Longer than openai's 600s default, which is below worst-case generation time at this budget.
CLIENT_TIMEOUT_SECONDS = 3600.0

_RGB_SCALES = {"g": (2, 6.0), "r": (1, 3.4), "z": (0, 2.2)}  # (channel, scale) per band
_BANDS = ["g", "r", "z"]


def _to_rgb(image: np.ndarray, m: float = 0.03, Q: int = 20) -> np.ndarray:
    """Raw nanomaggie flux -> normalized [0,1] RGB via arcsinh stretch, ported from astroclip.astrodino.data.augmentations.ToRGB."""
    channels = np.transpose(image, (2, 0, 1))  # (3, H, W)

    intensity = 0
    for channel, band in zip(channels, _BANDS):
        _, scale = _RGB_SCALES[band]
        intensity = intensity + np.maximum(0, channel * scale + m)
    intensity /= len(_BANDS)

    stretch = np.arcsinh(Q * intensity) / np.sqrt(Q)
    intensity = intensity + (intensity == 0.0) * 1e-6

    h, w = intensity.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for channel, band in zip(channels, _BANDS):
        plane, scale = _RGB_SCALES[band]
        rgb[:, :, plane] = (channel * scale + m) * stretch / intensity

    return np.clip(rgb, 0, 1)


def _image_to_data_url(image: np.ndarray) -> str:
    rgb_uint8 = (_to_rgb(image) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb_uint8).save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def parse_redshift(text: str) -> float | None:
    matches = re.findall(r"-?\d+\.?\d*", text)
    return float(matches[-1]) if matches else None


def _ask(client: OpenAI, model: str, data_url: str, max_tokens: int, thinking: bool):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QUESTION},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    return response.choices[0]


def _process_one(client: OpenAI, model: str, path: Path, thinking: bool) -> tuple[str, float, bool]:
    image = np.load(path)
    data_url = _image_to_data_url(image)

    choice = _ask(client, model, data_url, MAX_TOKENS, thinking)
    truncated = choice.finish_reason == "length"

    redshift = parse_redshift(choice.message.content or "")
    return path.stem, redshift if redshift is not None else 0.0, truncated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N query galaxies (for a quick check)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="concurrent requests; match the server's --max-num-seqs",
    )
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode (default: off)")
    args = parser.parse_args()

    query_paths = sorted((DATA_DIR / "query_images").glob("*.npy"))
    if args.limit:
        query_paths = query_paths[: args.limit]
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=CLIENT_TIMEOUT_SECONDS)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "thinking" if args.thinking else "no_thinking"
    output_path = RESULTS_DIR / f"backbone_prediction_{suffix}.csv"

    still_truncated = []
    failed = []
    with output_path.open("w", newline="") as f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        writer = csv.writer(f)
        writer.writerow(["targetid", "redshift"])
        future_to_path = {pool.submit(_process_one, client, args.model, path, args.thinking): path for path in query_paths}
        # Written as each item completes, so one failure keeps earlier predictions.
        for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="backbone"):
            path = future_to_path[future]
            try:
                targetid, redshift, truncated = future.result()
            except Exception as exc:
                print(f"WARNING: {path.stem} failed ({exc!r}), skipping", file=sys.stderr)
                failed.append(path.stem)
                continue
            writer.writerow([targetid, redshift])
            f.flush()
            if truncated:
                still_truncated.append(targetid)

    if still_truncated:
        print(f"WARNING: {len(still_truncated)} items still truncated even at {MAX_TOKENS} tokens: {still_truncated}")
    if failed:
        print(f"WARNING: {len(failed)} items failed and were skipped: {failed}")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
