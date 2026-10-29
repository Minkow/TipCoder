#!/usr/bin/env bash
set -euo pipefail
: "${POLICY_MODEL:?Set POLICY_MODEL to the SFT proposer}"
: "${PROMPT_DATA:?Set PROMPT_DATA to the RL dataset}"
: "${REMOTE_RM_URL:?Set REMOTE_RM_URL to the reward server /score endpoint}"
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
IFS= read -r -d '' TEMPLATE < "$ROOT/inference/proposer_template.txt" || true
exec python -m openrlhf.cli.train_ppo_ray \
    --pretrain "$POLICY_MODEL" --prompt_data "$PROMPT_DATA" \
    --input_key question --input_template "$TEMPLATE" \
    --remote_rm_url "$REMOTE_RM_URL" \
    --actor_num_nodes 1 --actor_num_gpus_per_node "${ACTOR_GPUS:-4}" \
    --ref_num_nodes 1 --ref_num_gpus_per_node "${ACTOR_GPUS:-4}" \
    --colocate_actor_ref --vllm_num_engines 1 \
    --vllm_tensor_parallel_size "${ROLLOUT_TP:-1}" \
    --micro_train_batch_size 8 --train_batch_size 256 \
    --micro_rollout_batch_size 8 --rollout_batch_size 512 \
    --n_samples_per_prompt 8 --advantage_estimator group_norm \
    --max_epochs 1 --num_episodes 1 --prompt_max_len 4096 --generate_max_len 4096 \
    --zero_stage 3 --bf16 --actor_learning_rate 1e-6 --init_kl_coef 1e-3 \
    --adam_offload --gradient_checkpointing --packing_samples --normalize_reward \
    --save_steps 5 --save_path "${OUTPUT_DIR:-outputs/rl}/model" \
    --ckpt_path "${OUTPUT_DIR:-outputs/rl}/checkpoints" "$@"
