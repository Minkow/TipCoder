# Code release review

Reviewed on 2026-10-30. Scope: the code-only release directory, including comments
and docstrings, not the private experimental workspace or model/data contents.

No embedded access tokens, private keys, internal IPv4 endpoints, personal
filesystem paths, or email addresses were found by the final text scan.
No result-backfilling/cherry-picking scripts are included in this release.
These checks are not a guarantee that arbitrary inputs or dependencies are safe.

## Experimental remnants

- The experience maker contains an old last-sample `###NO_TIPS###` replacement
  experiment. A literal `False` disables it even when its environment variable
  is set. Its misleading comment was corrected; executable behavior is unchanged.
- Three commented-out BCB prompt templates were removed. The active template
  and generation logic are unchanged.
- Training-time function-name repair, assertion filtering, prompt-prefix lookup,
  fallback rewards, and truncation remain as implemented. These heuristics are
  not hidden fixes or new behavior added by the release cleanup.

## Execution and network boundaries

- Generated programs and tests are executed by the training evaluator. Run on
  isolated workers without credentials or sensitive mounts.
- The RL data formatter uses `eval` on its input field. Only trusted input data
  is supported by this unchanged implementation.
- Model loaders use `trust_remote_code`; inspect model repositories before use.
- The reward HTTP service is not an authenticated public service. Keep it on a
  private interface or tunnel.
- Teacher APIs receive the prompts supplied by the caller. Dataset builders can
  upload to Hugging Face when explicitly run; review the destination first.

No GPU training or benchmark rerun is part of this release review. Code is
prepared as a private GitHub draft before any model/data publication.
