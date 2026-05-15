#!/bin/bash
# Stop the vLLM server launched by start_vllm.sh.

PORT=8765
PIDS=$(pgrep -f "vllm serve.*--port $PORT")

if [ -z "$PIDS" ]; then
  echo "vLLM not running on port $PORT"
  exit 0
fi

echo "stopping vLLM PIDs: $PIDS"
kill $PIDS

for _ in $(seq 1 10); do
  sleep 1
  pgrep -f "vllm serve.*--port $PORT" > /dev/null || { echo "stopped"; exit 0; }
done

echo "still alive after 10s, sending SIGKILL"
kill -9 $PIDS 2>/dev/null
