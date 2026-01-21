#!/bin/bash
# Codespace keepalive script
# Prevents idle timeout by generating periodic activity
# Run in background: nohup bash scripts/keepalive.sh &

echo "Starting Codespace keepalive..."
echo "PID: $$"
echo "Log file: /tmp/keepalive.log"

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] Keepalive ping" | tee -a /tmp/keepalive.log
    sleep 300  # 5 minutes
done
