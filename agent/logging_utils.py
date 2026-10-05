import json
import time
from pathlib import Path
from typing import Any

STDOUT_TRACE_PATH = Path("-")
TRACE_STDOUT_PREFIX = "__AUTO_RESEARCH_TRACE__ "


class TraceLogger:
    def __init__(self, trace_path: Path):
        self._file = None
        if trace_path != STDOUT_TRACE_PATH:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = trace_path.open("a", buffering=1)

    def log(self, event_type: str, **fields: Any) -> None:
        record = {"ts": time.time(), "event": event_type, **fields}
        line = json.dumps(record, default=str)
        if self._file is None:
            print(TRACE_STDOUT_PREFIX + line, flush=True)
        else:
            self._file.write(line + "\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
