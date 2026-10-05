#!/usr/bin/env bash
set -euo pipefail
pkill -f "vllm serve" && echo "Stopped vLLM server" || echo "vLLM server not running"
