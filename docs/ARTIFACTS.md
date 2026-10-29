# Datasets and Checkpoints

Code is hosted on GitHub; datasets and weights will be hosted separately on
Hugging Face. **The artifacts below are not yet available for download.**
Links will be added after upload has been verified.

## Training Data

Only the final SFT and RL training datasets are selected for release. Their
source files were identified from historical training task arguments, rather
than inferred from filenames. Internal version labels are not release names.

| Planned release file | Stage | Records | Source size |
| --- | --- | ---: | ---: |
| `sft/train.json` | Supervised tip-proposer training | 2,126 | 5.40 MB |
| `rl/train.json` | Reinforcement learning | 6,410 | 113.15 MB |

SFT records contain `id`, `source`, `prompt`, `output` and `pass_rate`.
RL records contain `id`, `source`, `question`, `test_cases`, `inferences` and
`context_messages`. No samples are removed or rewritten for the release.
Sizes above describe the source files; serialization may change the file size.

The datasets derive from [AceCode-87K](https://huggingface.co/datasets/TIGER-Lab/AceCode-87K).
The planned dataset release follows its MIT license with upstream attribution.
Treat generated code and tests as untrusted input; execute only in an isolated
environment.

## Proposer Checkpoints

The selected SFT checkpoint was trained from Qwen3-4B-Instruct-2507. The RL run
starts from that SFT checkpoint; the selected RL export is from step 10.
Historical directory labels do not describe the model architecture reliably.

The prepared RL export uses BF16 safetensors, tokenizer files and model
configuration. Its 399 saved tensors were checked byte-for-byte against the
source tensors after conversion to BF16. The package is approximately 8.84 GB,
including tokenizer files. This checks serialization, not generation parity.

Optimizer states, training logs, platform metadata and the original redundant
FP32 storage are excluded. Original checkpoints remain unchanged. Model-card
attribution and redistribution terms are separate from this repository's code
license. A verified download link will be added after publication.
