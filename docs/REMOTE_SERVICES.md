# Remote service boundaries

There are three distinct services. They are not interchangeable.

| Service | Client | Interface |
| --- | --- | --- |
| Proposer/coder model serving | Inference scripts | OpenAI-compatible `/v1/completions` or `/v1/chat/completions` |
| RL execution reward | Modified OpenRLHF | HTTP `POST /score`, `GET /health` |
| BigCodeBench execution | `bigcodebench.evaluate` | Gradio endpoint configured with `--gradio_endpoint` |

## RL reward server

Install the patched OpenRLHF environment on the server. It supplies reward
transport serialization and the original training-case execution evaluator. The retained
`sync_rm_service.py` calls `LocalRMActor` and waits for each request's result.
The actor loads the frozen code generator using vLLM, combines problem text and
proposer tips, executes generated code against training tests, and returns
execution-derived rewards. It uses all visible GPUs for tensor parallelism.

```bash
export GEN_MODEL_PATH=/path/to/code-generator
export POLICY_MODEL_PATH=/path/to/proposer-tokenizer
export RM_DATASET=/path/to/rl.json
export RM_INPUT_KEY=context_messages
export RM_GT_KEY=test_cases
export RM_BASELINE_MODEL=qwen_coder_2.5
export RM_USE_BASELINE=1
export RM_LENGTH_PENALTY=0.005
python training/reward_server/sync_rm_service.py --host 127.0.0.1 --port 8002
curl --fail http://localhost:8002/health
```

Requests contain equally sized `queries`, `prompts`, and `labels` lists.
The legacy `query` key is accepted. Responses contain the custom reward
payload serialized by `openrlhf.utils.rm_serialization`; use the patched client
so arrays and additional logs are decoded correctly. Keep the dataset used by
the server aligned with training prompts and IDs. Server-side baseline scores
are read from the dataset's `inferences` entries, not from AceCodeRM.

For a remote machine use a private route or SSH tunnel, for example:

```bash
ssh -N -L 8002:127.0.0.1:8002 user@reward-host
```

The code-execution service is not an authenticated public API or a secure
sandbox. Run it in an isolated worker/container without credentials or sensitive
mounts, and keep it reachable only by the training clients. Do not expose it
directly to the Internet. No automatic proxy, firewall, or network changes are
performed by these scripts.

## BigCodeBench remote evaluation

The historical workflow generated samples on GPU workers, then passed the saved
JSONL to a separate BigCodeBench execution server. It did not require generating
code again during evaluation. The server must have the BCB task dependencies
and a compatible Gradio API installed.

```bash
ssh -N -L 7860:127.0.0.1:7860 user@bcb-host
GRADIO_ENDPOINT=http://localhost:7860 \
  bash evaluation/bigcodebench_remote.sh outputs/selected.jsonl
```

This command retains the original `instruct full --execution gradio` workflow.
Use a BigCodeBench installation exposing `--gradio_endpoint`; check
`bigcodebench.evaluate --help` before running. The original BCB checkout did not
retain Git metadata, so its exact upstream revision has not been identified.
The server implementation and its dependency image are not included in this
first cleanup. They must be packaged separately before claiming turnkey remote
BCB deployment. Connection failures are infrastructure failures, not model
test failures; keep generated files for a later retry.
