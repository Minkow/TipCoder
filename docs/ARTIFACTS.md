# Datasets and Checkpoints

Code is hosted on GitHub. The BF16 RL proposer is available at
[Minkow/TipCoder-RL](https://huggingface.co/Minkow/TipCoder-RL).
The final training data is available at
[Minkow/TipCoder-Data](https://huggingface.co/datasets/Minkow/TipCoder-Data).

## Training Data

Only the final SFT and RL training datasets are selected for release. Their
source files were identified from historical training task arguments, rather
than inferred from filenames. Internal version labels are not release names.

| Release file | Stage | Records | File size |
| --- | --- | ---: | ---: |
| `sft/train.json` | Supervised tip-proposer training | 2,126 | 5.40 MB |
| `rl/train.json` | Reinforcement learning | 6,410 | 113.15 MB |

SFT records contain `id`, `source`, `prompt`, `output` and `pass_rate`.
RL records contain `id`, `source`, `question`, `test_cases`, `inferences` and
`context_messages`. No samples are removed or rewritten for the release.
The published files are byte-identical to the source files. SFT uses a JSON
array; RL retains JSON Lines serialization and its original `.json` suffix.

```bash
hf download Minkow/TipCoder-Data --repo-type dataset --local-dir data/TipCoder
```

```python
from datasets import load_dataset

sft = load_dataset("Minkow/TipCoder-Data", "sft", split="train")
rl = load_dataset("Minkow/TipCoder-Data", "rl", split="train")
```

The datasets derive from [AceCode-87K](https://huggingface.co/datasets/TIGER-Lab/AceCode-87K).
The dataset release follows its MIT license with upstream attribution.
Treat generated code and tests as untrusted input; execute only in an isolated
environment.

## Proposer Checkpoints

```bash
pip install -U huggingface_hub
hf download Minkow/TipCoder-RL --local-dir checkpoints/TipCoder-RL
```

The public checkpoint contains four BF16 safetensors shards, their index,
configuration/tokenizer files, a model card and license notices. Uploaded file
sizes and hashes were verified against the export. Verified model revision:
`66cf49845d6f147d3556703ff336eb6ea9b54f45`.

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
license. The upstream base-model Apache-2.0 license is included in the model
repository's LICENSE alongside the TIPCoder contribution notice.
