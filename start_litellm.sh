#!/bin/bash
# Launch LiteLLM proxy on port 4000, fronting vLLM at localhost:8765.
# Exposes Anthropic /v1/messages so Claude Code can drive the local model.
# Point Claude Code at it:
#   ANTHROPIC_BASE_URL=http://localhost:4000 ANTHROPIC_AUTH_TOKEN=dummy claude
# Logs to ~/.tui-chat/litellm.log. Prints PID; kill with `kill <PID>`.
set -e

PORT=4000
CONFIG=$(dirname "$(readlink -f "$0")")/litellm.config.yaml
LOG=$HOME/.tui-chat/litellm.log

if curl -sf http://localhost:$PORT/v1/models -m 2 > /dev/null 2>&1; then
  echo "LiteLLM already running on port $PORT"
  pgrep -f "litellm.*--port $PORT" | head -1
  exit 0
fi

mkdir -p "$(dirname "$LOG")"

nohup $HOME/miniconda3/envs/vllm/bin/litellm \
  --config "$CONFIG" \
  --port $PORT \
  > "$LOG" 2>&1 &

PID=$!
echo "LiteLLM starting (PID $PID), logs: $LOG"
echo "Ready when:  curl -sf http://localhost:$PORT/v1/models"
echo "Kill with:   kill $PID"
