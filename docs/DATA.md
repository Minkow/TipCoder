# Data construction

Data construction is kept separate from training. It produces files consumed
by the SFT/RL launchers; it never starts a training job. The retained builders
come from the existing training workspace rather than the early RiCoder format
conversion experiments.

| Script | Responsibility |
| --- | --- |
| `get_hard_data.py` | Select difficult examples from existing inference records |
| `build_sft_dataset.py` | Construct examples using code candidates and test outcomes |
| `build_reflection_sft.py` | Teacher repair trajectories with execution feedback |
| `build_compression_sft.py` | Distill instruction guidance from repair trajectories and verify it |
| `format_code_rl_data.py` | Format records for the RL data loader |

Run each script with `--help` to inspect its named Fire arguments. Inputs are
the AceCode-style records expected by the original scripts; arbitrary instruction
JSONL is not interchangeable. Preserve question text, IDs, test cases and
inference/pass-rate fields used to calculate baseline reward.

`get_hard_data.py` requires `--dataset_path`. Set `REFLECTION_PARTIAL_PATH`
and `REFLECTION_DATA_PATH` to the reflection artifacts to exclude, in that
priority order. These replace private absolute paths; the original filtering
and random-sampling behavior is retained, without a newly imposed seed.

The repair and compression builders still import the original training-case
evaluator from the patched OpenRLHF installation. Follow `TRAINING.md` to
install it, plus the root requirements and `datasets`. Directory separation
does not imply that these legacy runtime dependencies have been rewritten.

Provide teacher credentials through `OPENAI_API_KEY`. Pass `--api_base` and
`--openai_model` explicitly for the selected provider. The compression builder
also takes `--vllm_url` and `--vllm_model` for code verification. All model and
data paths belong to the caller; no private model or provider is distributed.

Generated training data and third-party dataset licenses require a separate
release decision. This code release contains no data.
