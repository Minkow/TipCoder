#!/usr/bin/env bash
set -euo pipefail
: "${POLICY_MODEL:?Set POLICY_MODEL to the proposer model}"
: "${SFT_DATA:?Set SFT_DATA to the prepared dataset}"
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
IFS= read -r -d '' TEMPLATE < "$ROOT/inference/proposer_template.txt" || true
exec deepspeed --module openrlhf.cli.train_sft \
    --pretrain "$POLICY_MODEL" --dataset "$SFT_DATA" \
    --input_key "${INPUT_KEY:-prompt}" --output_key "${OUTPUT_KEY:-output}" \
    --input_template "$TEMPLATE" \
    --save_path "${OUTPUT_DIR:-outputs/sft}" \
    --max_len "${MAX_LENGTH:-8192}" --train_batch_size 32 \
    --micro_train_batch_size 4 --max_epochs 1 --zero_stage 2 \
    --bf16 --learning_rate 5e-6 --gradient_checkpointing "$@"
