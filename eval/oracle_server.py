"""Host-side localhost server behind oracle_check; the reference solution never enters the container."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from eval.evaluate import evaluate

# Never return anchors or the model-choice verdict, which would leak grading details.
_ORACLE_RESULT_FIELDS = ("valid", "score", "metric", "n", "errors")


class OracleServer:
    """Serves one run on 127.0.0.1. workspace_dir is the host path of /agent_run/workspace."""

    def __init__(self, task: str, workspace_dir: Path):
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(task, workspace_dir))
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address
        return f"http://{host}:{port}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def _make_handler(task: str, workspace_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass  # orchestrate.py's own stdout stays free of per-request noise

        def do_POST(self) -> None:
            if self.path != "/check":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                relative_path = Path(body["relative_path"])
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(f"relative_path must be relative, within the workspace, got {body['relative_path']!r}")
                result = evaluate(task, workspace_dir / relative_path)
            except Exception as exc:
                result = {"valid": False, "errors": [f"{type(exc).__name__}: {exc}"]}
            payload = json.dumps({k: result.get(k) for k in _ORACLE_RESULT_FIELDS}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler
