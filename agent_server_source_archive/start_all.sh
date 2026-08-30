#!/bin/bash
# Start server (router + ASR + LLM) and voice agent in separate tmux sessions.
# Usage: ./start_all.sh [MODEL]
#        tmux attach -t server   (view server logs)
#        tmux attach -t agent    (view agent logs)
#        ./stop_all.sh           (kill everything)

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/env310/bin/python}"
MODEL="${1:-qwen3.5-4b-history}"
case "$MODEL" in
  *[!A-Za-z0-9._-]*) echo "ERROR: Invalid model name: $MODEL"; exit 1 ;;
esac

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python interpreter not executable: $PYTHON_BIN"
    exit 1
fi

echo "=== Starting model servers (router + ASR + LLM) ==="
tmux new-session -d -s server -c "$ROOT/server" \
  "cd $ROOT/server && $PYTHON_BIN manage.py start 2>&1; exec bash"
echo "  tmux session 'server' created. Attach: tmux attach -t server"

# Wait for models to be ready
echo "  Waiting for models to load..."
sleep 5
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:8000/health | grep -q '"ok"'; then
        echo "  Router is ready."
        break
    fi
    sleep 2
done

echo ""
echo "=== Starting Voice Agent ==="
tmux new-session -d -s agent -c "$ROOT" \
  "cd $ROOT && $PYTHON_BIN -m agents.Headless.voice_agent --host 0.0.0.0 --port 8766 --model $MODEL 2>&1; exec bash"
echo "  tmux session 'agent' created. Attach: tmux attach -t agent"

echo ""
echo "=== Done ==="
echo "  Frontend:  http://localhost:8000/frontend"
echo "  Agent debugger: http://localhost:8766/"
echo "  Agent API: http://localhost:8766/health"
echo "  Chat model: $MODEL"
echo ""
echo "  View server logs:  tmux attach -t server"
echo "  View agent logs:   tmux attach -t agent"
echo "  Detach from tmux:  Ctrl+B then D"
echo "  Stop everything:   $ROOT/stop_all.sh"
