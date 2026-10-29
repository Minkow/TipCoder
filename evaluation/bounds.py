"""Summarize aligned, single-candidate benchmark branches without dropping tasks."""

import argparse
import json
from pathlib import Path


def load_outcomes(path, metric):
    records = json.loads(Path(path).read_text())["eval"]
    if not records:
        raise ValueError(f"Empty evaluation: {path}")
    outcomes = {}
    for task, rows in records.items():
        if len(rows) != 1:
            raise ValueError(f"{task}: expected one candidate per branch in {path}")
        row = rows[0]
        fields = {
            "base": ("base_status",),
            "plus": ("base_status", "plus_status"),
            "bcb": ("status",),
        }[metric]
        if any(field not in row for field in fields):
            raise ValueError(f"{task}: missing {metric} status in {path}")
        outcomes[task] = all(row[field] == "pass" for field in fields)
    return outcomes


def summarize(base, guided, selected=None):
    if not base or not guided:
        raise ValueError("Nonempty base and at least one guided branch are required")
    branches = [base, *guided]
    for branch in branches + ([selected] if selected is not None else []):
        if set(branch) != set(base):
            raise ValueError("Task sets differ; no samples may be dropped")
    tasks = list(base)

    def pct(values):
        return 100 * sum(values) / len(tasks)

    result = {
        "n_tasks": len(tasks),
        "base": pct(base.values()),
        "raw_by_branch": [pct(branch.values()) for branch in guided],
        "lower": pct(all(branch[t] for branch in branches) for t in tasks),
        "upper": pct(any(branch[t] for branch in branches) for t in tasks),
    }
    if selected is not None:
        result["rm"] = pct(selected.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=["base", "plus", "bcb"], required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--guided", nargs="+", required=True)
    parser.add_argument("--selected", help="Evaluation of the RM-selected samples")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(
        load_outcomes(args.base, args.metric),
        [load_outcomes(path, args.metric) for path in args.guided],
        load_outcomes(args.selected, args.metric) if args.selected else None,
    )
    output = Path(args.output)
    if output.resolve() in {
        Path(p).resolve() for p in [args.base, *args.guided, args.selected] if p
    }:
        raise ValueError("Output must not overwrite an input")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
