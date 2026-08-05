#!/bin/bash
set -e

echo "========================================"
echo "Starting HGAST Docker Space"
echo "========================================"

echo "[1/4] Starting FastAPI backend..."

cd /app/backend

PYTHONPATH=/app/backend \
uvicorn hgast_framework.hgast_v2_api:app \
    --host 0.0.0.0 \
    --port 7861 &

echo "[2/4] Waiting for backend..."
sleep 15

echo "[3/4] Starting Next.js frontend..."

cd /app/frontend

echo "========================================"
echo "HGAST is ready!"
echo "Frontend : http://0.0.0.0:7860"
echo "Backend  : http://0.0.0.0:7861"
echo "========================================"

exec env HOSTNAME=0.0.0.0 PORT=7860 npm start
