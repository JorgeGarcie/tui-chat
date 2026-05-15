#!/bin/bash
# Stop the LiteLLM proxy launched by start_litellm.sh.

PORT=4000
PIDS=$(pgrep -f "litellm.*--port $PORT")

if [ -z "$PIDS" ]; then
  echo "LiteLLM not running on port $PORT"
  exit 0
fi

echo "stopping LiteLLM PIDs: $PIDS"
kill $PIDS

for _ in $(seq 1 10); do
  sleep 1
  pgrep -f "litellm.*--port $PORT" > /dev/null || { echo "stopped"; exit 0; }
done

echo "still alive after 10s, sending SIGKILL"
kill -9 $PIDS 2>/dev/null
