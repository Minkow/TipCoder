# Dataset and checkpoint release inventory

Metadata inspected on 2026-10-30. No datasets or weights are included in Git.
Internal source paths are stored only in the maintainer's private inventory.
The names below identify files, not a promise that every candidate will be released.

## Checkpoints

| Candidate | Located export | Config architecture | Stored size | Status |
| --- | --- | --- | --- | --- |
| SFT | `AceCode-SFT-q5` | Qwen3ForCausalLM | 7.51 GiB | Two safetensors shards, index references present |
| RL | `global_step10/converted_hf` in the `0123` run | Qwen3ForCausalLM | 59.95 GiB | Four bin shards; investigate storage overhead before packaging |

These paths were selected by the retained fullprompt experiment launchers.
Directory names are historical labels and must not be used to infer architecture.
Only configuration and shard presence were inspected; tensor loading, base-model
identity, redistribution terms and final artifact sanitization remain pending.

The RL index declares 16,089,872,384 bytes of tensor data while its files occupy
about 64.4 GB. Do not rename another checkpoint or convert dtype to work around
this. Inspect storage and create a separate export only after checking tensor
equivalence. Exclude optimizer states, training logs, RNG state and platform metadata.

## Dataset candidates

| File | Records | Approximate file size | Status |
| --- | ---: | ---: | --- |
| `acecode_87k_final_sft_v1_filtered_clean.json` | 4,274 | 5.29 MB | SFT candidate |
| `acecode_87k_final_sft_v1_filtered_clean_v3.json` | 3,974 | 5.00 MB | Alternate SFT candidate |
| `acecode_compression_sft.json` | 2,968 | 15.57 MB | Distillation intermediate |
| `acecode_87k_v6.json` | 13,342 | 230.39 MB | RL candidate |

Record counts were read from actual files, including JSONL stored with a `.json`
suffix. No records were modified or discarded. Final selection, full content
privacy review and source-dataset redistribution terms are not yet confirmed.

## Distribution

Keep code on GitHub. Publish approved model and dataset artifacts separately,
with model/dataset cards and links here. Do not apply the code's Apache-2.0
license automatically to weights, upstream examples or teacher-generated data.
No model or dataset hosting repository has been created by this preparation.
