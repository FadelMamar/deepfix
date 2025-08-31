#!/usr/bin/env bash

# MLflow Tracking Server Launch Script
# This script launches the MLflow tracking server using uv

# Set default values
HOST=${MLFLOW_HOST:-0.0.0.0}
PORT=${MLFLOW_PORT:-5000}
BACKEND_STORE_URI=${MLFLOW_BACKEND_STORE_URI:-sqlite:///mlflow.db}
DEFAULT_ARTIFACT_ROOT=${MLFLOW_DEFAULT_ARTIFACT_ROOT:-./mlruns}

echo "Starting MLflow Tracking Server..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Backend Store URI: $BACKEND_STORE_URI"
echo "Default Artifact Root: $DEFAULT_ARTIFACT_ROOT"

# Launch MLflow tracking server using uv
uv run mlflow server \
    --host "$HOST" \
    --port "$PORT" \
    --backend-store-uri "$BACKEND_STORE_URI" \
    --default-artifact-root "$DEFAULT_ARTIFACT_ROOT"
