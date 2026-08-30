#!/bin/bash
# Start only the voice agent in a tmux session.
# Prerequisite: model servers must be running (start_all.sh or manage.py start).
#
# Usage: ./start_agent.sh [MODEL]
#        tmux attach -t agent   (view logs)
#        Ctrl+B then D          (detach)

ROOT="$(cd "$(dirname "$0")" && pwd)"
SESSION="agent"
PYTHON_BIN="${PYTHON_BIN:-/data/env310/bin/python}"
MODEL="${1:-qwen3.5-4b-history}"
case "$MODEL" in
  *[!A-Za-z0-9._-]*) echo "ERROR: Invalid model name: $MODEL"; exit 1 ;;
esac

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python interpreter not executable: $PYTHON_BIN"
    exit 1
fi

# Check router health
if ! curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "ERROR: Router (port 8000) not reachable."
    echo "   Start with: cd $ROOT/server && python manage.py start"
    exit 1
fi

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null

# Start voice agent in tmux
tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "cd $ROOT && MEDICAL_DENSE_ENABLED=1 $PYTHON_BIN -m agents.Headless.voice_agent --host 0.0.0.0 --port 8766 --model $MODEL 2>&1; exec bash"

echo "✓ Voice agent started (session: $SESSION)"
echo ""
echo "  Attach:  tmux attach -t $SESSION"
echo "  Detach:  Ctrl+B then D"
echo "  URL:     ws://<host>:8766/ws"
echo "  Debug:   http://<host>:8766/"
echo "  Model:   $MODEL"
echo "  Health:  curl http://<host>:8766/health"
echo "  Stop:    tmux kill-session -t $SESSION"
