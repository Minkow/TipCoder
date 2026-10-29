#!/usr/bin/env bash
set -euo pipefail
: "${GRADIO_ENDPOINT:?Set GRADIO_ENDPOINT to your BigCodeBench execution service}"
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 samples.jsonl" >&2
    exit 1
fi
exec python -m tipcoder_eval bcb --subset full --samples "$1" \
    --endpoint "$GRADIO_ENDPOINT"
