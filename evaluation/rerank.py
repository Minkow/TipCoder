"""Select one candidate per task using the original AceCodeRM/InternLM scorers."""

import argparse
import json
import math
from pathlib import Path

from reward_models import load_ace_rm, load_internlm_rm, score_ace, score_internlm


def load_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_candidates(rows, scores):
    if len(rows) != len(scores):
        raise ValueError("One reward score is required for every candidate")
    chosen = {}
    for row, score in zip(rows, scores):
        if not math.isfinite(score):
            raise ValueError(f"Non-finite reward for {row['task_id']}")
        task = row["task_id"]
        # Strict comparison preserves the first (base) candidate on ties.
        if task not in chosen or score > chosen[task]["reward_score"]:
            chosen[task] = dict(row, reward_score=score)
    return list(chosen.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, help="Original benchmark JSONL")
    parser.add_argument(
        "--candidates",
        required=True,
        nargs="+",
        help="Native file first, then guided files",
    )
    parser.add_argument(
        "--prompt-field", default="prompt", help="Use instruct_prompt for BCB"
    )
    parser.add_argument("--rm-model", required=True)
    parser.add_argument("--rm-kind", choices=["ace", "internlm"], default="ace")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prompt_rows = load_rows(args.prompts)
    prompts = {r["task_id"]: r[args.prompt_field] for r in prompt_rows}
    if len(prompts) != len(prompt_rows):
        raise ValueError("Duplicate task IDs in prompt file")
    rows = []
    for index, path in enumerate(args.candidates):
        branch = load_rows(path)
        if {r["task_id"] for r in branch} != set(prompts):
            raise ValueError(f"Candidate task IDs differ from prompts: {path}")
        for row in branch:
            code = row.get("solution", row.get("completion"))
            if not isinstance(code, str):
                raise ValueError("Each candidate requires solution or completion text")
            rows.append(dict(row, solution=code, candidate_branch=index))
    load, score = (
        (load_ace_rm, score_ace)
        if args.rm_kind == "ace"
        else (load_internlm_rm, score_internlm)
    )
    tokenizer, model = load(args.rm_model)
    scores = score(
        tokenizer,
        model,
        [prompts[r["task_id"]] for r in rows],
        [r["solution"] for r in rows],
        args.batch_size,
        args.max_length,
    )
    selected = select_candidates(rows, scores)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected)
    )
    output.with_suffix(".scores.jsonl").write_text(
        "".join(
            json.dumps(dict(r, reward_score=s), ensure_ascii=False) + "\n"
            for r, s in zip(rows, scores)
        )
    )


if __name__ == "__main__":
    main()
