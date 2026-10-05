"""Tests Slurm dispatch of hosted-API tiers: no vLLM server job, correct --dependency chaining.

sbatch is stubbed; created spec, run and log files are removed by each test.
"""

import shutil

import pytest
import yaml

from orchestrate import REPO_ROOT
from scripts.production import continue_grid_slurm as continue_mod
from scripts.production import run_grid_slurm as run_mod

WHOLE_NODE_PROFILE = {
    "_CLUSTER_NAME": "test-whole-node",
    "WHOLE_NODE_ONLY": "1",
    "GPUS_PER_NODE": "4",
    "CORES_PER_NODE": "64",
    "GRES_TEMPLATE": "gpu:{n}",
    "SERVER_QOS": "normal",
    "WORKER_QOS": "normal",
    "TEARDOWN_PARTITION": "gpu",
    "TEARDOWN_QOS": "normal",
    "SERVER_NEEDS_INTERNET_FLAG": "--comment=INTERNET_ACCESS=1",
    "WORKER_NEEDS_INTERNET_FLAG": "--comment=INTERNET_ACCESS=1",
}


class _FakeSbatch:
    """Records every call, returns incrementing fake job ids -- never touches a real scheduler."""

    def __init__(self):
        self.calls: list[tuple[list[str], str, list[str]]] = []
        self._next_id = 1000

    def __call__(self, extra_args, script, script_args, env=None):
        self._next_id += 1
        self.calls.append((extra_args, script, script_args))
        return str(self._next_id)


def _write_spec(name: str, model_levels: list[str]) -> None:
    spec = {
        "task": "mmlu-astronomy",
        "axes": {
            "information": ["none"],
            "harness": ["react"],
            "verification": ["asked"],
            "budget": ["short"],
            "model": model_levels,
        },
        "replicates": 1,
    }
    (REPO_ROOT / "specs" / f"{name}.yaml").write_text(yaml.safe_dump(spec))


def _cleanup(name: str) -> None:
    (REPO_ROOT / "specs" / f"{name}.yaml").unlink(missing_ok=True)
    for grid_dir in (REPO_ROOT / "runs").glob(f"{name}-*"):
        shutil.rmtree(grid_dir, ignore_errors=True)
    for log_dir in (REPO_ROOT / "slurm" / "logs").glob(f"{name}-*"):
        shutil.rmtree(log_dir, ignore_errors=True)


@pytest.fixture
def spec_name(request):
    name = f"_test-{request.node.name}".replace("[", "-").replace("]", "")
    yield name
    _cleanup(name)


def test_single_hosted_api_tier_submits_no_server_or_teardown(spec_name, monkeypatch):
    _write_spec(spec_name, ["claude-sonnet-5"])
    fake_sbatch = _FakeSbatch()
    monkeypatch.setattr(run_mod, "sbatch", fake_sbatch)

    rc = run_mod.run_grid_slurm(
        spec_name, partition="gpu", server_qos="normal", worker_qos="normal",
        max_concurrent=4, profile=WHOLE_NODE_PROFILE,
    )

    assert rc == 0
    scripts_called = [call[1] for call in fake_sbatch.calls]
    assert scripts_called == ["agent_worker_array_packed.sbatch"], (
        "a hosted-API tier must submit exactly one job (its worker array) -- "
        f"no vllm_server.sbatch or vllm_teardown.sbatch, got {scripts_called}"
    )
    _extra_args, _script, worker_args = fake_sbatch.calls[0]
    assert worker_args[3] == "none", "server-job-id positional arg must be the 'none' sentinel"


def test_first_tier_worker_array_has_no_dependency_flag(spec_name, monkeypatch):
    """Nothing to wait for -- must not depend on the literal string 'none'."""
    _write_spec(spec_name, ["claude-sonnet-5"])
    fake_sbatch = _FakeSbatch()
    monkeypatch.setattr(run_mod, "sbatch", fake_sbatch)

    run_mod.run_grid_slurm(
        spec_name, partition="gpu", server_qos="normal", worker_qos="normal",
        max_concurrent=4, profile=WHOLE_NODE_PROFILE,
    )

    extra_args, _script, _worker_args = fake_sbatch.calls[0]
    dependency_args = [a for a in extra_args if a.startswith("--dependency=")]
    assert dependency_args == [], f"first tier should have no --dependency at all, got {dependency_args}"


def test_after_flag_reaches_a_hosted_api_first_tier(spec_name, monkeypatch):
    _write_spec(spec_name, ["claude-sonnet-5"])
    fake_sbatch = _FakeSbatch()
    monkeypatch.setattr(run_mod, "sbatch", fake_sbatch)

    run_mod.run_grid_slurm(
        spec_name, partition="gpu", server_qos="normal", worker_qos="normal",
        max_concurrent=4, profile=WHOLE_NODE_PROFILE, after="42",
    )

    extra_args, _script, _worker_args = fake_sbatch.calls[0]
    assert "--dependency=afterok:42" in extra_args


def test_vllm_tier_then_hosted_api_tier_chains_afterok_off_teardown(spec_name, monkeypatch):
    _write_spec(spec_name, ["qwen35-35b-a3b-fp8", "claude-sonnet-5"])
    fake_sbatch = _FakeSbatch()
    monkeypatch.setattr(run_mod, "sbatch", fake_sbatch)

    rc = run_mod.run_grid_slurm(
        spec_name, partition="gpu", server_qos="normal", worker_qos="normal",
        max_concurrent=4, profile=WHOLE_NODE_PROFILE,
    )

    assert rc == 0
    scripts_called = [call[1] for call in fake_sbatch.calls]
    # tier 1 (vLLM): server + worker array + teardown; tier 2 (hosted API): worker array only.
    assert scripts_called == [
        "vllm_server.sbatch", "agent_worker_array_packed.sbatch", "vllm_teardown.sbatch",
        "agent_worker_array_packed.sbatch",
    ]
    # Tier 2's (hosted-API) worker array must depend on tier 1's teardown succeeding (afterok) --
    # the same chaining a second vLLM tier would get, just off a worker array instead of a server.
    tier2_extra_args = fake_sbatch.calls[3][0]
    dependency_args = [a for a in tier2_extra_args if a.startswith("--dependency=")]
    assert len(dependency_args) == 1
    assert dependency_args[0].startswith("--dependency=afterok:")


def test_hosted_api_tier_then_vllm_tier_chains_afterany_off_worker_array(spec_name, monkeypatch):
    _write_spec(spec_name, ["claude-sonnet-5", "step37-198b-a11b-fp8"])
    fake_sbatch = _FakeSbatch()
    monkeypatch.setattr(run_mod, "sbatch", fake_sbatch)

    rc = run_mod.run_grid_slurm(
        spec_name, partition="gpu", server_qos="normal", worker_qos="normal",
        max_concurrent=4, profile=WHOLE_NODE_PROFILE,
    )

    assert rc == 0
    scripts_called = [call[1] for call in fake_sbatch.calls]
    # tier 1 (hosted API): worker array only; tier 2 (vLLM): server + worker array + teardown.
    assert scripts_called == [
        "agent_worker_array_packed.sbatch", "vllm_server.sbatch",
        "agent_worker_array_packed.sbatch", "vllm_teardown.sbatch",
    ]
    tier1_array_job = "1001"  # _FakeSbatch's first returned id (1000 + 1)
    server_extra_args = fake_sbatch.calls[1][0]
    dependency_args = [a for a in server_extra_args if a.startswith("--dependency=")]
    assert dependency_args == [f"--dependency=afterany:{tier1_array_job}"]


def test_preemptible_is_rejected_for_a_hosted_api_tier(spec_name, monkeypatch):
    _write_spec(spec_name, ["claude-sonnet-5"])
    fake_sbatch = _FakeSbatch()
    monkeypatch.setattr(run_mod, "sbatch", fake_sbatch)

    with pytest.raises(SystemExit, match="preemptible"):
        run_mod.run_grid_slurm(
            spec_name, partition="gpu", server_qos="normal", worker_qos="normal",
            max_concurrent=4, profile=WHOLE_NODE_PROFILE, preemptible=True,
        )
    assert fake_sbatch.calls == []


def test_continue_grid_slurm_also_skips_server_for_a_hosted_api_tier(spec_name, monkeypatch, tmp_path):
    """continue_grid_slurm duplicates run_grid_slurm's per-tier logic."""
    _write_spec(spec_name, ["claude-sonnet-5"])
    grid_dir_name = f"{spec_name}-resume-test"
    grid_dir = REPO_ROOT / "runs" / grid_dir_name
    grid_dir.mkdir(parents=True)
    try:
        # One cell, manifest written but no score.json -- "remaining" ends up covering it.
        manifest_path = grid_dir / ".manifest-claude-sonnet-5.tsv"
        manifest_path.write_text("none\treact\tasked\tshort\tclaude-sonnet-5\t1\n")

        fake_sbatch = _FakeSbatch()
        monkeypatch.setattr(continue_mod, "sbatch", fake_sbatch)

        rc = continue_mod.continue_grid_slurm(
            grid_dir_name, spec_name, partition="gpu", server_qos="normal", worker_qos="normal",
            max_concurrent=4, profile=WHOLE_NODE_PROFILE,
        )

        assert rc == 0
        scripts_called = [call[1] for call in fake_sbatch.calls]
        assert scripts_called == ["agent_worker_array_packed.sbatch"]
        worker_args = fake_sbatch.calls[0][2]
        assert worker_args[3] == "none"
    finally:
        shutil.rmtree(grid_dir, ignore_errors=True)
