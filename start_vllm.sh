#!/bin/bash
# Launch vLLM serving Qwen3-Coder-Next on port 8765, TP=2 across A6000s.
# Logs to ~/.tui-chat/vllm.log. Prints PID; kill with `kill <PID>`.
set -e

PORT=8765
MODEL=cyankiwi/Qwen3-Coder-Next-AWQ-4bit
LOG=$HOME/.tui-chat/vllm.log

if curl -sf http://localhost:$PORT/v1/models -m 2 > /dev/null 2>&1; then
  echo "vLLM already running on port $PORT"
  pgrep -f "vllm serve.*--port $PORT" | head -1
  exit 0
fi

mkdir -p "$(dirname "$LOG")"

PATH=$HOME/miniconda3/envs/vllm/bin:$PATH \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1 \
nohup $HOME/miniconda3/envs/vllm/bin/vllm serve "$MODEL" \
  --port $PORT \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  > "$LOG" 2>&1 &

PID=$!
echo "vLLM starting (PID $PID), logs: $LOG"
echo "Ready when:  curl -sf http://localhost:$PORT/v1/models"
echo "Kill with:   kill $PID"
