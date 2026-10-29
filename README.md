# TIPCoder

Code for training a test-time instruction proposer and selecting between native
and tip-guided code candidates. This tree is a code-cleanup draft for
[TipCoder (Findings of EMNLP 2026)](https://aclanthology.org/2026.findings-emnlp.495/).

Original contributions use Apache-2.0; see `LICENSE`, `NOTICE` and
`THIRD_PARTY.md`. Datasets and checkpoints are not yet published. Their current
packaging status is documented in [ARTIFACTS.md](docs/ARTIFACTS.md).

## Layout

| Directory | Purpose |
| --- | --- |
| `data_construction/` | Hard-example filtering, repair trajectories, masked tip distillation, RL data formatting |
| `training/` | SFT/RL launchers and the modified OpenRLHF integration |
| `training/reward_server/` | Remote code rollout and execution-reward service for RL |
| `inference/` | Existing tip generation scripts and an optional code-generation adapter |
| `evaluation/` | Benchmark execution wrappers, RM scorers, optional reranking/bounds utilities |
| `docs/` | Setup, remote service boundaries, retained scripts and dependencies |

No datasets, model weights, experiment outputs, cluster launchers or credentials
are included. Supply model and dataset locations explicitly. GPU communication
uses the runtime defaults; launchers do not disable P2P, shared memory or IB,
bind a cluster network interface, or configure a tracking proxy.

## Inference

The tip generators retain their original processing behavior. The chat coder
adapter and generic reranking/bounds utilities below were added during cleanup;
they are optional examples, not replacements for the deployed experiment
pipeline. See [cleanup boundaries](docs/CLEANUP.md) before choosing an entry point.

Use Python 3.10 or newer. Install the lightweight clients with
`python -m pip install -r requirements.txt`, then
`python -m pip install -e ./evaluation` for the independent eval module.
Install vLLM in a separate GPU environment
to serve the proposer and coder via OpenAI-compatible endpoints. Model versions
and GPU-library dependencies must be selected for your installed CUDA stack.

```bash
vllm serve /path/to/proposer --served-model-name proposer --port 8000
```

The proposer requires `/v1/completions` (preformatted training prompt), while
the portable coder client uses `/v1/chat/completions`. Serve the coder on a
separate port/GPU allocation; its endpoint must support `seed`.

Export the original benchmark records to JSONL, preserving task IDs and fields.
For EvalPlus this can be done using `get_human_eval_plus` or `get_mbpp_plus`
from `evalplus.data`. Then:

```bash
python inference/generate_evalplus_tips.py --dataset humaneval \
  --cached-file data/humaneval.jsonl --api-base http://localhost:8000/v1 \
  --output outputs/humaneval_tips.jsonl

python inference/generate_code.py --input data/humaneval.jsonl \
  --api-base http://localhost:8001/v1 --model coder --output outputs/native.jsonl
python inference/generate_code.py --input outputs/humaneval_tips.jsonl \
  --api-base http://localhost:8001/v1 --model coder --output outputs/guided.jsonl

python -m pip install -r evaluation/requirements-rm.txt
python evaluation/rerank.py --prompts data/humaneval.jsonl \
  --candidates outputs/native.jsonl outputs/guided.jsonl \
  --rm-model /path/to/AceCodeRM --rm-kind ace --output outputs/selected.jsonl
bash evaluation/evalplus.sh humaneval outputs/selected.jsonl
```

For MBPP use `--dataset mbpp`. For BCB generate the override records with
`inference/generate_bcb_tips.py`; use `--prompt-field instruct_prompt` for coder
generation and reranking. The BCB script resolves relative output paths under
`--local-cache-dir`; an absolute `--output` avoids ambiguity. Run
`evaluation/bigcodebench_remote.sh` to evaluate the selected code remotely.

For the BCB generator, configure `--tokenizer-models` (or `QWEN_CODER_MODEL`
and `DS_CODER_MODEL`) when using its original length guard. To match a run
that disabled this guard, pass `--max-prompt-tokens 0` explicitly; the script's
original default is 2048. No new tips sanitization or retry policy is added.

The new `generate_code.py` is a portable chat-API adapter, not the historical
EvalPlus/BCB fork's generation launcher. It extracts complete solutions using
EvalPlus sanitization. It does not silently truncate input prompts. Choose a
context window and output budget supported by your coder. Reranking consumes
already-cleaned code; it never uses execution outcomes to choose candidates.

See [training](docs/TRAINING.md), [data construction](docs/DATA.md), and
[remote services](docs/REMOTE_SERVICES.md). This cleanup has not been validated
by a new GPU training run; environment compatibility remains to be checked.
