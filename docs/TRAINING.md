# Training integration

The development checkout identifies itself as OpenRLHF 0.9.0 and is based on
commit `cf69bd61a5fbcfa4f07bad2af76797643ac557e5`. Do not substitute the latest
OpenRLHF release: the modifications depend on this revision's trainer APIs.

The release uses a pinned upstream checkout plus an overlay, rather than a
second vendored framework. `training/install_openrlhf.sh` creates a fresh clone,
checks out that exact commit, and applies the source overlay. It refuses to
overwrite an existing checkout. Review modifications with `git diff` in the
prepared checkout; the new evaluator/serialization files are untracked there.

```bash
bash training/install_openrlhf.sh
python -m pip install -e vendor/OpenRLHF
python -m pip install -e ./evaluation
```

`training/openrlhf-requirements.txt` records the development checkout's dependency
declarations, including DeepSpeed 0.18.0, Transformers 4.57.0 and Ray 2.48.0.
These declarations are not a complete environment lock. The code execution
modules also require EvalPlus, NumPy, psutil, termcolor and tqdm; the reward
server requires vLLM and datasets. Install GPU dependencies in a dedicated
Linux environment appropriate for the host's CUDA and driver versions.

## Modified modules

- `datasets/prompts_dataset.py`: retain original prompts alongside formatted inputs.
- `trainer/ppo_utils/experience_maker.py`: propagate prompt information and handle custom rewards.
- `trainer/ppo_trainer.py`, `trainer/ray/ppo_actor.py`: custom reward/training integration.
- `utils/remote_rm_utils.py`: decode and merge remote reward responses.
- `utils/rm_serialization.py`: transport reward arrays/tensors.
- `cli/train_ppo_ray.py`: training CLI integration.
- `cli/eval_test_cases.py`, `cli/evalplus_eval.py`: original training-test execution.

Training execution remains in the overlay with its original behavior. The
separate `evaluation/` directory is for RiCoder benchmark evaluation, not a
replacement for training-test execution.

Upstream copyright/license is retained in `training/OPENRLHF_LICENSE`.
Project files derived from AceCoder require its attribution as described in
`THIRD_PARTY.md`. Original project contributions use MIT; third-party
source retains its existing notices and licensing.

## SFT

The launcher takes JSON/JSONL records with `prompt` and `output` fields by
default. Override `INPUT_KEY`/`OUTPUT_KEY` for another prepared schema.
`inference/proposer_template.txt` supplies the training prompt.

```bash
POLICY_MODEL=/path/to/base-model SFT_DATA=/path/to/sft.jsonl \
  OUTPUT_DIR=outputs/sft bash training/train_sft.sh
```

## RL

Start the execution-reward server described in `REMOTE_SERVICES.md` first.
The proposer rollout and actor/reference training run in OpenRLHF/Ray;
the remote server owns the code generator and execution workers. The latter
is separate from inference-time AceCodeRM/InternLM reranking.

```bash
POLICY_MODEL=/path/to/sft-model PROMPT_DATA=/path/to/rl.json \
  REMOTE_RM_URL=http://localhost:8002/score OUTPUT_DIR=outputs/rl \
  ACTOR_GPUS=4 ROLLOUT_TP=1 bash training/train_rl.sh
```

Default topology colocates actor/reference on four GPUs and requests one
additional proposer rollout engine. The reward server needs its own allocation.
Configure Ray resources and GPU visibility to fit the deployment; no GPU count
or NCCL workaround is inferred from the original cluster. Additional CLI
arguments are forwarded to OpenRLHF. No jobs or GPU servers are started by the
installation script.

Communication cleanup removes the launcher's P2P/SHM overrides and the copied
actor's forced `NCCL_CUMEM_ENABLE=0`. The latter was a vLLM synchronization
compatibility workaround. Test synchronization with the chosen runtime before
a long training run; the release no longer forces it for every environment.
