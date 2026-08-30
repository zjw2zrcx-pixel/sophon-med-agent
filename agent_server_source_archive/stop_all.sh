#!/bin/bash
# Stop all services started by start_all.sh

echo "=== Stopping Voice Agent ==="
tmux send-keys -t agent C-c 2>/dev/null || true
sleep 1
tmux kill-session -t agent 2>/dev/null || true
echo "  Agent session killed."

echo "=== Stopping model servers ==="
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/server" && python manage.py stop 2>/dev/null || true
sleep 2
tmux send-keys -t server C-c 2>/dev/null || true
tmux kill-session -t server 2>/dev/null || true
echo "  Server session killed."

echo "=== All stopped ==="
