"""Generate code from original or tip-augmented JSONL through a model endpoint."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.input).read_text().splitlines()
        if line.strip()
    ]
    client = OpenAI(
        base_url=args.api_base, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY")
    )

    def generate(row):
        prompt = row[args.prompt_field]
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": "Please provide a self-contained Python script that solves the following problem in a markdown code block:\n"
                    + prompt,
                }
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        raw = response.choices[0].message.content
        if not raw:
            raise ValueError(f"Empty generation for {row['task_id']}")
        from tipcoder_eval.code import sanitize

        return {
            "task_id": row["task_id"],
            "solution": sanitize(raw, entrypoint=row.get("entry_point")),
            "raw_solution": raw,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        generated = list(
            tqdm(pool.map(generate, rows), total=len(rows), desc="Code generation")
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generated)
    )


if __name__ == "__main__":
    main()
