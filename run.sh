#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/app"
set -e

mkdir -p /app/log
mkdir -p /app/shared/experiments

echo "--- Directory Check ---"
echo "Current Directory: $(pwd)"
echo "Contents of /app:"
ls -F

echo "Contents of /app/shared (Mounted Volume):"
ls -F shared/

echo "--- Starting Training ---"
python main.py