"""The agent's tools, each defined once as a JSON schema and a system-prompt bullet."""

import html
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import trafilatura

from agent.config import is_verified

MAX_READ_FILE_CHARS = 4000
# Larger than read_file's cap, since a web page cannot be read in slices.
MAX_WEB_FETCH_CHARS = 20000
WEB_FETCH_TIMEOUT_S = 15
# Several sites block httpx's default user agent.
WEB_FETCH_USER_AGENT = "auto-research-agent/0.1 (autonomous research agent MVP)"

# Always blocked: loopback and cloud metadata hosts.
DEFAULT_WEB_DENYLIST = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254"})

# Env allowlist for run_bash, so the agent's code cannot read its condition labels, the backbone
# URL or ORACLE_URL.
BASH_ENV_PASSTHROUGH = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "LANG", "LC_ALL", "TMPDIR",
    "PYTHONUNBUFFERED", "UV_CACHE_DIR", "PIP_REQUIRE_VIRTUALENV",
    "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
})
# The harness's own venv, removed from run_bash's PATH.
FRAMEWORK_VENV_BIN = "/app/.venv/bin"
# Default run_bash venv baked into agent.sif (agent/base_env/pyproject.toml).
BASE_ENV_VENV = "/opt/agent-base-env/.venv"
BASE_ENV_BIN = f"{BASE_ENV_VENV}/bin"


@dataclass(frozen=True)
class ToolSpec:
    """A tool's JSON schema and its system-prompt bullet."""

    schema: dict[str, Any]
    prompt: str

    @property
    def name(self) -> str:
        return self.schema["function"]["name"]


TOOLS: list[ToolSpec] = [
    ToolSpec(
        prompt="- read_file(path): read a file. You may read anything under /task (the task "
        "description and any provided data, read-only), /models (any models mounted for this task, "
        "read-only -- if present, each has its own subdirectory with a README describing what it "
        "is -- read it), or /agent_run/workspace (your own scratch directory). Output is truncated "
        "for large files.",
        schema={
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text file. Allowed under /task (read-only), /models (read-only, "
                "any models mounted for this task -- see their own README for what they are and how "
                "to use them), or /agent_run/workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Absolute path to read"}},
                    "required": ["path"],
                },
            },
        },
    ),
    ToolSpec(
        prompt="- write_file(path, content): write a file. Only paths under /agent_run/workspace "
        "are writable.",
        schema={
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write a UTF-8 text file. Only allowed under /agent_run/workspace. "
                "Overwrites the file if it already exists; creates parent directories as needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute path to write, under /agent_run/workspace"},
                        "content": {"type": "string", "description": "File content"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
    ),
    ToolSpec(
        prompt="- run_bash(command): run a shell command with working directory /agent_run/workspace. "
        "A Python env with numpy, pandas, scikit-learn, scipy, torch, transformers, and accelerate is "
        "already active by default -- just `import` them, no install needed. For anything beyond that, "
        "`uv pip install <package>` adds it to this same env (a shared uv package cache makes this "
        "fast); there is no system `pip` and no writable system site-packages, so always use `uv pip "
        "install`, never plain `pip install`. If you want a fully separate env instead of extending the "
        "default one, `uv venv` + `source .venv/bin/activate` in the same command (each run_bash call "
        "starts fresh, so re-activate every time you need it). "
        "Any model under /models/<name> is a separate matter: it already "
        "has its own complete, pre-built Python environment, entirely unrelated to your workspace's -- do "
        "NOT `uv run`/`uv venv`/`uv pip install` against it or expect it to show up in whatever environment "
        "you build yourself, no matter your current directory. Instead, invoke its own interpreter directly "
        "by absolute path exactly as its README says (typically `/models/<name>/env/.venv/bin/python "
        "your_script.py`) -- that binary already has everything that model needs installed. Do not use this "
        "tool for internet access (e.g. curl/wget) -- use web_fetch instead.",
        schema={
            "type": "function",
            "function": {
                "name": "run_bash",
                "description": "Run a shell command with cwd=/agent_run/workspace. Returns stdout, stderr, and exit code.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": "Shell command to execute"}},
                    "required": ["command"],
                },
            },
        },
    ),
    ToolSpec(
        prompt="- web_fetch(url): fetch an http(s) URL and get back its main text content (ads/nav/"
        "boilerplate stripped out), with links kept as [text](url) so you can find further URLs to fetch. "
        "Use this to research a topic, look up library documentation, check a current fact, or find "
        "anything else you need from the internet -- there is no separate search tool, so to search, fetch "
        "a search engine's results page directly, e.g. https://lite.duckduckgo.com/lite/?q=your+query, then "
        "fetch whichever result URL looks relevant. Some hosts may be blocked; a blocked or failed request "
        "returns an error explaining why instead of stopping the run.",
        schema={
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "Fetch a URL over HTTP(S) and return its main text content, with links kept "
                "as [text](url) (truncated if large). Redirects are not followed automatically -- a "
                "redirect response reports its target so it can be fetched explicitly. Subject to an "
                "allow/deny list; a blocked request returns an error explaining why instead of raising.",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string", "description": "Absolute http:// or https:// URL"}},
                    "required": ["url"],
                },
            },
        },
    ),
]


# Oracle ablation: scores a submission against the reference. Enabled by Config.oracle_url,
# independent of the verification level.
ORACLE_TOOL = ToolSpec(
    prompt="- oracle_check(submission_path): score a submission file against this task's reference "
    "solution and get back your real performance on it, in the same units finish()'s expected_score "
    "would report. Does not end the run -- call finish separately, whenever you're ready. Call this "
    "as many times as you like.",
    schema={
        "type": "function",
        "function": {
            "name": "oracle_check",
            "description": "Score a submission file against this task's reference solution and "
            "return the resulting metric value. Does not end the run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "submission_path": {
                        "type": "string",
                        "description": "Path under /agent_run/workspace to the submission to score",
                    },
                },
                "required": ["submission_path"],
            },
        },
    },
)


def active_tools(oracle_enabled: bool) -> list[ToolSpec]:
    """TOOLS plus ORACLE_TOOL if enabled; the single source for prompt and schemas."""
    return [*TOOLS, ORACLE_TOOL] if oracle_enabled else list(TOOLS)


@dataclass(frozen=True)
class FinishParameter:
    """One finish() parameter; `description` serves as both schema text and prompt sentence."""

    name: str
    json_type: str
    description: str
    verified_only: bool = False  # verified levels only; a lower one never offers it (agent/config.py's is_verified)
    required: bool = True


FINISH_PARAMETERS: list[FinishParameter] = [
    FinishParameter(
        "submission_path",
        "string",
        "submission_path must be a path under /agent_run/workspace to your final output (whatever "
        "artifact the task asks for -- a predictions file, a report, a script, etc).",
    ),
    FinishParameter(
        "summary",
        "string",
        "summary should briefly describe your approach and findings.",
    ),
    FinishParameter(
        "model_used",
        "string",
        'If you decide to use one of the models under /models to produce your output, set '
        'model_used to its directory name (e.g. "astroclip"); if you decided not to use any of '
        "them, omit it.",
        required=False,
    ),
    FinishParameter(
        "expected_score",
        "number",
        "expected_score is your own numeric estimate of your output's performance on this task's "
        "stated metric.",
        verified_only=True,
    ),
    FinishParameter(
        "verification_evidence",
        "string",
        "verification_evidence is a short description of what you actually did to check your output "
        "before submitting (what you compared it against, what you ran to sanity-check it) -- not "
        "just a restated intention to verify.",
        verified_only=True,
    ),
]


def finish_parameters(level: str) -> list[FinishParameter]:
    """The finish() parameters this verification level offers, in declaration order."""
    verified = is_verified(level)
    return [p for p in FINISH_PARAMETERS if verified or not p.verified_only]


def finish_schema(level: str) -> dict[str, Any]:
    parameters = finish_parameters(level)
    return {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Signal that the task is complete and the final output is ready.",
            "parameters": {
                "type": "object",
                "properties": {p.name: {"type": p.json_type, "description": p.description} for p in parameters},
                "required": [p.name for p in parameters if p.required],
            },
        },
    }


def build_tool_schemas(level: str, oracle_enabled: bool = False) -> list[dict[str, Any]]:
    """Every active tool's schema plus a finish() schema that varies by verification level."""
    return [tool.schema for tool in active_tools(oracle_enabled)] + [finish_schema(level)]


class SandboxViolation(Exception):
    pass


@dataclass(frozen=True)
class RawToolCall:
    """Backend-agnostic tool call. `arguments` is the raw string, so parse errors reach the model."""

    id: str
    name: str
    arguments: str


@dataclass
class FinishSignal:
    submission_path: str
    summary: str
    model_used: str | None = None
    # Only offered by the verified levels.
    expected_score: float | None = None
    verification_evidence: str | None = None


class Tools:
    def __init__(
        self,
        task_dir: Path,
        workspace_dir: Path,
        bash_timeout_s: int,
        verification: str,
        models_dir: Path = Path("/models"),
        web_allowlist: frozenset[str] = frozenset(),
        web_denylist: frozenset[str] = frozenset(),
        oracle_url: str | None = None,
    ):
        # Dispatch drops finish() parameters this verification level does not offer.
        self.finish_parameters = frozenset(p.name for p in finish_parameters(verification))
        self.task_dir = task_dir.resolve()
        self.workspace_dir = workspace_dir.resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        # May not exist if the task mounts no models.
        self.models_dir = models_dir.resolve()
        self.bash_timeout_s = bash_timeout_s
        self.web_allowlist = web_allowlist
        self.web_denylist = web_denylist | DEFAULT_WEB_DENYLIST
        # Host-side eval/oracle_server.py; None if oracle_check is not granted.
        self.oracle_url = oracle_url

    def _resolve(self, path: str, roots: tuple[Path, ...], what: str) -> Path:
        """Resolve (symlinks included) and require containment in one of `roots`."""
        resolved = (self.workspace_dir / path).resolve() if not path.startswith("/") else Path(path).resolve()
        if any(resolved == root or root in resolved.parents for root in roots):
            return resolved
        raise SandboxViolation(f"{path} is outside the allowed {what} ({', '.join(str(r) for r in roots)})")

    def _resolve_readable(self, path: str) -> Path:
        return self._resolve(path, (self.task_dir, self.models_dir, self.workspace_dir), "read roots")

    def _resolve_writable(self, path: str) -> Path:
        return self._resolve(path, (self.workspace_dir,), "write root")

    def read_file(self, path: str) -> str:
        resolved = self._resolve_readable(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"No such file: {resolved}")
        content = resolved.read_text(errors="replace")
        if len(content) > MAX_READ_FILE_CHARS:
            omitted = len(content) - MAX_READ_FILE_CHARS
            content = (
                content[:MAX_READ_FILE_CHARS]
                + f"\n\n[... truncated, {omitted} more characters omitted. This file is large -- "
                "use run_bash with `head`/`wc -l`/pandas to inspect it instead of reading it whole ...]"
            )
        return content

    def write_file(self, path: str, content: str) -> str:
        resolved = self._resolve_writable(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        return f"Wrote {len(content)} bytes to {resolved}"

    def run_bash(self, command: str) -> str:
        # Models sometimes HTML-escape shell operators ("&&" -> "&amp;&amp;").
        command = html.unescape(command)
        env = {k: v for k, v in os.environ.items() if k in BASH_ENV_PASSTHROUGH}
        # Strip the harness venv so the agent's code uses its own environment.
        env["PATH"] = ":".join(p for p in env.get("PATH", "").split(":") if p and p != FRAMEWORK_VENV_BIN)
        # Set in the command, since the login shell's /etc/profile overwrites PATH from env=.
        command = f'export PATH="{BASE_ENV_BIN}:$PATH" VIRTUAL_ENV="{BASE_ENV_VENV}"\n{command}'
        try:
            result = subprocess.run(
                ["bash", "-lc", command],
                cwd=self.workspace_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.bash_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return f"[timed out after {self.bash_timeout_s}s]"
        return (
            f"exit_code: {result.returncode}\n"
            f"stdout:\n{result.stdout[-8000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )

    def _check_web_access(self, host: str) -> None:
        host = host.lower()
        if any(host == d or host.endswith("." + d) for d in self.web_denylist):
            raise SandboxViolation(f"{host} is on the web denylist")
        if self.web_allowlist and not any(host == d or host.endswith("." + d) for d in self.web_allowlist):
            raise SandboxViolation(f"{host} is not on the web allowlist ({sorted(self.web_allowlist)})")

    def web_fetch(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"url must start with http:// or https://, got {url!r}")
        host = urlparse(url).hostname
        if not host:
            raise ValueError(f"could not parse a hostname from {url!r}")
        self._check_web_access(host)

        try:
            response = httpx.get(
                url,
                timeout=WEB_FETCH_TIMEOUT_S,
                follow_redirects=False,
                headers={"User-Agent": WEB_FETCH_USER_AGENT},
            )
        except httpx.HTTPError as exc:
            return f"error: request failed: {exc}"

        if response.is_redirect:
            location = response.headers.get("location", "?")
            return f"status: {response.status_code} (redirect, not followed) -> {location}"

        text = response.text
        if "html" in response.headers.get("content-type", ""):
            # Strip boilerplate but keep links for further fetching.
            extracted = trafilatura.extract(text, include_links=True, url=url)
            text = extracted if extracted else re.sub(r"<[^>]+>", " ", html.unescape(text)).strip()
        if len(text) > MAX_WEB_FETCH_CHARS:
            omitted = len(text) - MAX_WEB_FETCH_CHARS
            text = text[:MAX_WEB_FETCH_CHARS] + f"\n\n[... truncated, {omitted} more characters omitted ...]"
        return f"status: {response.status_code}\n{text}"

    def oracle_check(self, submission_path: str) -> str:
        if self.oracle_url is None:
            raise RuntimeError("oracle_check is not available for this run")
        resolved = self._resolve_writable(submission_path)
        if not resolved.is_file():
            raise FileNotFoundError(f"submission file not found: {resolved}")
        # Sends a workspace-relative path; the host-side server reads the same file.
        relative_path = str(resolved.relative_to(self.workspace_dir))
        try:
            response = httpx.post(f"{self.oracle_url}/check", json={"relative_path": relative_path}, timeout=60.0)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as exc:
            return f"error: oracle request failed: {exc}"
        if not result.get("valid"):
            return "Submission format INVALID:\n" + "\n".join(f"  - {e}" for e in result.get("errors", []))
        return f"{result['metric']}: {result['score']:.4f} ({result['n']} rows)"

    def finish(
        self,
        submission_path: str,
        summary: str,
        model_used: str | None = None,
        expected_score: float | None = None,
        verification_evidence: str | None = None,
    ) -> FinishSignal:
        resolved = self._resolve_writable(submission_path)
        if not resolved.is_file():
            raise FileNotFoundError(f"submission file not found: {resolved}")
        return FinishSignal(
            submission_path=str(resolved),
            summary=summary,
            model_used=model_used or None,
            expected_score=expected_score,
            verification_evidence=verification_evidence or None,
        )

    def dispatch(self, name: str, arguments: dict[str, Any]) -> tuple[str, FinishSignal | None]:
        if name == "read_file":
            return self.read_file(**arguments), None
        if name == "write_file":
            return self.write_file(**arguments), None
        if name == "run_bash":
            return self.run_bash(**arguments), None
        if name == "web_fetch":
            return self.web_fetch(**arguments), None
        if name == "oracle_check":
            return self.oracle_check(**arguments), None
        if name == "finish":
            signal = self.finish(**{k: v for k, v in arguments.items() if k in self.finish_parameters})
            return f"Task marked finished: {signal.submission_path}", signal
        raise ValueError(f"Unknown tool: {name}")
