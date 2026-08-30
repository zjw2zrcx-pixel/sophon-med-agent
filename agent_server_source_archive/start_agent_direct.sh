#!/bin/bash
# Start voice agent directly in the current shell (no tmux).
# Usage: ./start_agent_direct.sh [MODEL]
#        Ctrl+C to stop

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/env310/bin/python}"
cd "$ROOT"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python interpreter not executable: $PYTHON_BIN"
    exit 1
fi

echo "=== Voice Agent ==="
echo "  Host: 0.0.0.0  Port: 8766"
echo "  Hotword: 小麦"
echo "  Press Ctrl+C to stop"
echo ""

MODEL_ARGS=()
if [ -n "${1:-}" ]; then
    MODEL_ARGS=(--model "$1")
fi
"$PYTHON_BIN" -m agents.Headless.voice_agent --host 0.0.0.0 --port 8766 "${MODEL_ARGS[@]}"
