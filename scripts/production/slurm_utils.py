"""Shared Slurm helpers for run_grid_slurm, continue_grid_slurm (whole-grid resume) and
sweep_grid_slurm (retry of preempted cells). Resubmission only selects cells without score.json;
orchestrate.launch_run decides what to do with each run_dir.
"""

import subprocess
from pathlib import Path

from dotenv import dotenv_values

from axes import RunSpec
from orchestrate import REPO_ROOT

SLURM_DIR = REPO_ROOT / "slurm"
CLUSTER_DIR = SLURM_DIR / "clusters"


def load_cluster_profile() -> dict[str, str]:
    """Key/value pairs of slurm/clusters/<CLUSTER>.env (default example), used as CLI defaults."""
    root_env = dotenv_values(REPO_ROOT / ".env")
    cluster = (root_env.get("CLUSTER") or "example").strip()
    profile_path = CLUSTER_DIR / f"{cluster}.env"
    if not profile_path.is_file():
        raise SystemExit(f"Unknown CLUSTER={cluster!r} in .env -- no {profile_path}")
    profile = dict(dotenv_values(profile_path))
    profile["_CLUSTER_NAME"] = cluster
    return profile


def gres(profile: dict[str, str], n: int) -> str:
    return profile["GRES_TEMPLATE"].format(n=n)


def write_manifest(path: Path, cells: list[RunSpec]) -> None:
    """Write one tab-separated cell per line, in Slurm array task order."""
    lines = [
        "\t".join((c.information, c.harness, c.verification, c.budget, c.model, str(c.replicate)))
        for c in cells
    ]
    path.write_text("\n".join(lines) + "\n")


def sbatch(extra_args: list[str], script: str, script_args: list[str], env: dict | None = None) -> str:
    """Submits a job with --parsable, returns its job id."""
    result = subprocess.run(
        ["sbatch", "--parsable", *extra_args, str(SLURM_DIR / script), *script_args],
        capture_output=True, text=True, check=True, env=env,
    )
    # --parsable can print "jobid;cluster" depending on the Slurm build, and an array job as
    # "jobid_taskid" in some contexts -- the bare leading jobid is what --dependency/sacct want.
    return result.stdout.strip().split(";")[0].split("_")[0]


def remaining_indices(manifest_path: Path, tier_cells: list[RunSpec], grid_dir: Path) -> list[int]:
    """1-indexed manifest lines of cells without score.json; a scored cell is never retried."""
    manifest_lines = manifest_path.read_text().splitlines()
    if len(manifest_lines) != len(tier_cells):
        raise SystemExit(
            f"Manifest {manifest_path} has {len(manifest_lines)} lines but the spec expands this "
            f"tier to {len(tier_cells)} cells -- spec must have changed since this grid was "
            f"launched, can't safely align remaining cells."
        )
    return [
        i for i, cell in enumerate(tier_cells, start=1)
        if not (grid_dir / cell.run_name / "score.json").is_file()
    ]


def array_spec(indices: list[int], throttle: int) -> str:
    """Compresses a sorted list of 1-indexed task ids into Slurm's --array range syntax, e.g.
    [3, 5, 6, 7, 12] -> '3,5-7,12%<throttle>'."""
    if not indices:
        return ""
    parts = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = i
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return f"{','.join(parts)}%{throttle}"
