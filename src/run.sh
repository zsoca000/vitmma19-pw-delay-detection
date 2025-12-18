#!/bin/bash
set -e

echo "--- Directory Check ---"
echo "Current Directory: $(pwd)"
echo "Contents of /app:"
ls -F

echo "Contents of /app/data (Mounted Volume):"
ls -F data/

echo "--- Starting Training ---"
python src/train.py