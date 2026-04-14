#!/bin/sh
# Conditionally export ONNX model paths when exported model artifacts exist.
#
# When the image is built with EXPORT_ONNX_MODELS=true (default) the Dockerfile
# bakes ONNX model directories into /app/models. Setting these env vars lets
# EmbeddingService and SentimentService load via onnxruntime instead of
# falling back to torch. When EXPORT_ONNX_MODELS=false the directories are
# absent, the env vars stay unset, and services use their HF Hub fallback.
#
# Externally-set values are honoured (we only set when unset).
set -e

if [ -z "${EMBEDDING_ONNX_MODEL_PATH:-}" ] && [ -f /app/models/embedding-finbert/model.onnx ]; then
    export EMBEDDING_ONNX_MODEL_PATH=/app/models/embedding-finbert
fi
if [ -z "${EMBEDDING_ONNX_MINILM_MODEL_PATH:-}" ] && [ -f /app/models/embedding-minilm/model.onnx ]; then
    export EMBEDDING_ONNX_MINILM_MODEL_PATH=/app/models/embedding-minilm
fi
if [ -z "${SENTIMENT_ONNX_MODEL_PATH:-}" ] && [ -f /app/models/sentiment-finbert/model.onnx ]; then
    export SENTIMENT_ONNX_MODEL_PATH=/app/models/sentiment-finbert
fi

# Dispatch:
#   - bare CLI subcommands (worker, serve, ...) get prepended with `news-tracker`
#   - explicit interpreters/paths (sh, python, /bin/...) run as-is
case "$1" in
    backtest|cleanup|cluster|clustering-worker|daily-clustering|drift|embedding-worker|graph|health|ingest|init-db|narrative|process|run-once|sentiment-worker|serve|vector-search|worker)
        exec news-tracker "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
