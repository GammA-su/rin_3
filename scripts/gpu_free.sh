#!/usr/bin/env bash
set -euo pipefail

echo "[*] Checking GPU usage:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits || true
echo

stop_user_service() {
  local svc=$1
  if systemctl --user list-units | grep -q "${svc}.service"; then
    systemctl --user stop "$svc" || true
    systemctl --user disable "$svc" || true
    systemctl --user mask "$svc" || true
    echo "[*] User service '$svc' stopped and masked"
  fi
}

stop_system_service() {
  local svc=$1
  if systemctl list-units | grep -q "${svc}.service"; then
    sudo systemctl stop "$svc" || true
    sudo systemctl disable "$svc" || true
    sudo systemctl mask "$svc" || true
    echo "[*] System service '$svc' stopped and masked"
  fi
}

echo "[*] Stopping Ollama services (user/system) if present..."
stop_user_service ollama
stop_system_service ollama

echo "[*] Killing stray processes (ollama, lmstudio) if running..."
pkill -f 'ollama' 2>/dev/null || true
pkill -f 'lmstudio' 2>/dev/null || true

echo
echo "[*] Post-clean GPU usage:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits || true
echo "[✓] GPU free pass complete (services masked; re-enable with systemctl unmask/enable)."

