# RiCoder experimental evaluation

This directory contains the experimental evaluation stages previously spread
across RiCoder's benchmark launchers, reranking scripts and bound experiments:

- `evalplus.sh`: execute HumanEval/MBPP tests on generated samples.
- `bigcodebench_remote.sh`: send BCB samples to the remote execution service.
- `rerank.py` and `reward_models.py`: select candidates with AceCodeRM or InternLM.
- `bounds.py`: compute base/raw/lower/upper and selected accuracy from results.

Generation stays in `inference/`; evaluation consumes its saved outputs.
No training checkpoint or OpenRLHF installation is required for these stages.
RM inference has its own optional `requirements-rm.txt`.

The scorer functions come from the existing RiCoder implementation. The
generic `rerank.py` and `bounds.py` interfaces were added during cleanup and
are optional tools, not the original experiment orchestration scripts. They
must not silently replace existing launchers or change their candidate inputs.

After evaluating each branch and the selected samples:

```bash
python evaluation/bounds.py --metric plus \
  --base outputs/native_eval_results.json \
  --guided outputs/sft_eval_results.json outputs/rl_eval_results.json \
  --selected outputs/selected_eval_results.json --output outputs/bounds.json
```

Use `--metric bcb` for BigCodeBench, `base` for original EvalPlus tests,
and `plus` for the augmented tests. Each input must contain exactly one
candidate per task; multi-sample runs must first be split into candidate
branches. All branches must contain identical task IDs. Lower/upper mean
all/any branches pass, including native. Raw is reported separately per guided
branch rather than inventing one raw score for SFT+RL. Selection uses RM scores,
never execution outcomes. Evaluate the exact saved code that was scored by RM;
this module does not silently apply an additional code/tips cleaning pass.

```bash
python -m pip install -e ./evaluation
tipcoder-eval --help
```

This package provides benchmark evaluation entry points. It has no direct dependency on OpenRLHF,
Ray, DeepSpeed, vLLM or the proposer/reward-model checkpoints. EvalPlus and its
transitive dependencies are still required for execution. BigCodeBench is an
optional external installation for the BCB subcommand.

```bash
tipcoder-eval evalplus --dataset humaneval --samples outputs/selected.jsonl
tipcoder-eval evalplus --dataset mbpp --samples outputs/selected.jsonl
tipcoder-eval bcb --samples outputs/selected.jsonl --subset full \
  --endpoint http://localhost:7860
```

The benchmark commands delegate to the respective benchmark implementations.
They consume generated/selected solutions and do not load any generation or
reward model. Remote BCB setup is described in `../docs/REMOTE_SERVICES.md`.

The training-case evaluator remains at
`training/openrlhf_overlay/openrlhf/cli/eval_test_cases.py`, importing its
original `evalplus_eval.py` helper. Its entry-point repair, extraction, timeout
handling, cache behavior and worker behavior have not been redesigned. It must
not replace official benchmark tests. Execute untrusted code in isolated workers.
