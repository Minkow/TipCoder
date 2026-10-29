#!/usr/bin/env python3
"""Build EvalPlus override datasets using full-prompt proposer input and _r tips protocol.

Differences from rewrite_evalplus_instructions_r.py:
- proposer sees the full EvalPlus prompt, not the extracted docstring only;
- the active _r template is preserved: force tips after ###TIPS###, no NO_TIPS path;
- empty tips are treated as a failed run instead of silently falling back to native.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


TIP_TEMPLATE = (
    "<|im_start|>system\n"
    "You are an expert programming mentor. Analyze the following coding instruction first, "
    'followed by the separator "###TIPS###", and then provide specific tips to help '
    "developers implement this correctly.\n"
    "Do not provide solution or any solution-related ideas. Your only task is to provide tips "
    "to clarify the instruction and identify potential pitfalls, common mistakes, and key "
    "considerations during solving the problem.\n"
    "Your tips should be problem-specific, not general programming principles. Do not conclude "
    "debug/test/verify tips.<|im_end|>\n"
    "<|im_start|>user\n"
    "Instruction: {}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def load_cached_dataset(
    dataset: str, cached_file: str | None = None
) -> list[dict[str, Any]]:
    if cached_file:
        path = Path(cached_file)
    else:
        names = {
            "humaneval": "evalplus_humaneval_cached.jsonl",
            "mbpp": "evalplus_mbpp_cached.jsonl",
        }
        if dataset not in names:
            raise ValueError(f"unsupported dataset: {dataset}")
        candidates = [Path(names[dataset]), Path("data") / names[dataset]]
        path = next((p for p in candidates if p.exists()), candidates[-1])
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(
        json.dumps(
            {"dataset": dataset, "cache": str(path), "n": len(rows)}, ensure_ascii=False
        ),
        flush=True,
    )
    return rows


def clean_tip(text: str) -> str:
    tip = (text or "").strip()
    if "###TIPS###" in tip:
        tip = tip.split("###TIPS###")[-1].strip()
    if "[TIPS]" in tip:
        tip = tip.split("[TIPS]")[-1].strip()
    for marker in ("debugging conversation:", "on the analysis:"):
        if marker in tip:
            tip = tip.split(marker)[-1].strip()
    return tip.strip()


def indent_tip(tip: str, indent: str = "    ") -> str:
    return "".join(indent + line.rstrip() + "\n" for line in tip.splitlines())


def inject_humaneval(prompt: str, tip: str) -> str:
    addition = "\n" + indent_tip(tip, "    ")
    for quote in ('"""', "'''"):
        idx = prompt.rfind(quote)
        if idx != -1:
            return prompt[:idx].rstrip() + addition + prompt[idx:]
    return prompt.rstrip() + "\n" + tip.strip() + "\n"


def inject_mbpp(prompt: str, tip: str) -> str:
    addition = "\n" + tip.strip() + "\n"
    for quote in ('"""', "'''"):
        idx = prompt.rfind(quote)
        if idx != -1:
            return prompt[:idx].rstrip() + addition + prompt[idx:]
    return prompt.rstrip() + "\n" + tip.strip() + "\n"


def inject_prompt(dataset: str, prompt: str, tip: str) -> str:
    if dataset == "humaneval":
        return inject_humaneval(prompt, tip)
    if dataset == "mbpp":
        return inject_mbpp(prompt, tip)
    raise ValueError(f"unsupported dataset: {dataset}")


def normalize_api_base(api_base: str) -> str:
    api_base = api_base.rstrip("/")
    if api_base.endswith("/v1"):
        return api_base
    return api_base + "/v1"


def generate_tip(
    client: OpenAI,
    model_name: str,
    full_prompt: str,
    max_prompt_chars: int,
    max_tokens: int,
    temperature: float,
) -> str:
    prompt = TIP_TEMPLATE.format(full_prompt[:max_prompt_chars])
    last_error = None
    for attempt in range(4):
        try:
            response = client.completions.create(
                model=model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=["<|endoftext|>", "<|im_end|>", "<|im_start|>"],
            )
            tip = clean_tip(response.choices[0].text)
            if tip:
                return tip
            last_error = "empty tips"
        except Exception as exc:
            last_error = exc
        time.sleep(2 + attempt)
    raise RuntimeError(f"tip generation failed: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--cached-file", default=None)
    parser.add_argument("--mode", default="tips")
    parser.add_argument("--sample-only", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-tip-tokens", type=int, default=4096)
    parser.add_argument("--max-prompt-chars", type=int, default=12000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if args.mode != "tips":
        raise ValueError("make_evalplus_tips_fullprompt_r.py only supports --mode tips")

    rows = load_cached_dataset(args.dataset, args.cached_file)
    if args.sample_only:
        rows = rows[: args.sample_only]

    prompts = [row.get("prompt", "") for row in rows]
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=normalize_api_base(args.api_base),
    )
    model_name = client.models.list().data[0].id
    print(
        json.dumps(
            {
                "model": model_name,
                "dataset": args.dataset,
                "n": len(rows),
                "protocol": "full_prompt_r_force_tips",
                "mean_full_prompt_chars": sum(len(x) for x in prompts)
                / max(1, len(prompts)),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    def run(prompt: str) -> str:
        return generate_tip(
            client,
            model_name,
            prompt,
            args.max_prompt_chars,
            args.max_tip_tokens,
            args.temperature,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        tips = list(executor.map(run, prompts))

    empty = [row["task_id"] for row, tip in zip(rows, tips) if not tip.strip()]
    if empty:
        raise RuntimeError(
            f"empty generated tips are not allowed: {empty[:10]} total={len(empty)}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for row, prompt, tip in zip(rows, prompts, tips):
            out = dict(row)
            out["original_prompt"] = prompt
            out["original_instruction"] = prompt
            out["generated_tips"] = tip
            out["enhanced_instruction"] = prompt + "\n" + tip.strip()
            out["fullprompt_r_tip_prompt_source"] = "full_prompt"
            out["fullprompt_r_tip_input_chars"] = len(prompt)
            out["prompt"] = inject_prompt(args.dataset, prompt, tip)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "output": str(out_path),
                "n": len(rows),
                "nonempty_tips": sum(bool(t.strip()) for t in tips),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
