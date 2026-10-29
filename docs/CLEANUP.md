# Cleanup boundaries

This release preparation changes organization and deployment configuration,
not the established experimental or training behavior. The original development
checkout is untouched. Data, weights, outputs and credentials remain outside
the release tree.

## Retained implementations

- `make_evalplus_tips_fullprompt_r.py` -> `inference/generate_evalplus_tips.py`.
- `rewrite_bigcodebench_instructions_r_fullprompt.py` -> `inference/generate_bcb_tips.py`.
- The four AceCodeRM/InternLM loader/scorer functions from
  `stat_bounds_rm_evalplus_clean_seeded.py` -> `evaluation/reward_models.py`.
- Training data builders -> `data_construction/`.
- Execution reward services -> `training/reward_server/`.
- Modified OpenRLHF files, including the prompt dataset and original execution
  evaluators -> `training/openrlhf_overlay/`.

The BCB completion call, tip extraction, exception fallback and length-guard
default are retained from the original script. The cleanup's shared generator
replacement has been withdrawn. Its unused earlier `generate_tips_batch_v0`
implementation is omitted; the active function is retained.

The training evaluators are no longer extracted into `tipcoder_eval`. Their
original single/multi-worker logic, result caching, code repair and timeout
handling remain unchanged. Data construction retains its original random
sampling; the cleanup's imposed random seed and edge-case changes were removed.

Private paths and credentials are supplied by arguments/environment variables.
Cluster-specific CUDA communication overrides are removed at the author's
request. These deployment differences are intentional, not claims of identical
behavior across environments. See `TRAINING.md`, `DATA.md` and
`REMOTE_SERVICES.md` for configuration.

## Optional new tools

`inference/generate_code.py`, `evaluation/rerank.py`, `evaluation/bounds.py`
and the benchmark CLI wrappers were introduced during cleanup. They are
optional adapters, not the deployed experiment orchestration. The new chat
coder in particular does not reproduce the historical benchmark forks' code
generation. Existing workflows must not be switched to these tools implicitly.

The remote BCB server and benchmark fork packaging remain separate work;
the current tree does not claim turnkey installation of those deployments.

## Preserved quirks

The review identified prefix-based question matching, whole-batch zero rewards
on missing matches, string-based timeout configuration, direct dataset `eval`,
and heuristic training-code extraction. These are recorded, not repaired as
part of cleanup. Changing them requires a separate behavioral change decision.
Use isolated execution environments and trusted training inputs.

Verification here covers source-level preservation, syntax, CPU utility tests
and private-configuration scans. It is not a new training run, a reproduction
audit, or evidence of historical experiment correctness.
