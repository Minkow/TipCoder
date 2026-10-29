#!/usr/bin/env python3
"""
Rewrite BigCodeBench dataset instructions using a vLLM endpoint.
"""

import argparse
import json
import os
import sys
from tqdm import tqdm
from openai import OpenAI


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _count_prompt_tokens(prompt: str, tokenizer, instruction_prefix=None, response_prefix=None) -> int:
    """Count tokens for the prompt as evalplus' vLLM backend would see it.

    - If tokenizer has chat_template, evalplus wraps prompt via make_raw_chat_prompt.
    - Otherwise it's a direct completion.
    """
    try:
        if getattr(tokenizer, "chat_template", None):
            from evalplus.provider.utility import make_raw_chat_prompt

            wrapped = make_raw_chat_prompt(prompt, instruction_prefix, response_prefix, tokenizer)
            return len(tokenizer(wrapped, add_special_tokens=False).input_ids)
        return len(tokenizer(prompt, add_special_tokens=False).input_ids)
    except Exception:
        # Conservative fallback: approximate by character length.
        return len(prompt)


def _should_drop_tips(
    prompt_with_tips: str,
    prompt_without_tips: str,
    tokenizer_models: list[str],
    max_prompt_tokens: int,
    trust_remote_code: bool = False,
    instruction_prefix=None,
    response_prefix=None,
) -> bool:
    """Return True if we should drop tips to satisfy max_prompt_tokens.

    Rule: if ANY configured tokenizer model says prompt_with_tips exceeds the limit,
    we drop tips entirely (revert to prompt_without_tips). If no tokenizer models are
    available, we do not drop (but print a warning).
    """
    if max_prompt_tokens <= 0:
        return False

    if not tokenizer_models:
        return False

    from transformers import AutoTokenizer

    # Match evalplus defaults (see evalplus.codegen) so token counting is consistent.
    if instruction_prefix is None:
        instruction_prefix = os.getenv(
            "REWRITE_EVALPLUS_INSTRUCTION_PREFIX",
            "Please provide a self-contained Python script that solves the following problem in a markdown code block:",
        )
    if response_prefix is None:
        response_prefix = os.getenv(
            "REWRITE_EVALPLUS_RESPONSE_PREFIX",
            "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:",
        )

    available = 0
    for model_name in tokenizer_models:
        try:
            tok = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            print(f"[length-guard] skip tokenizer '{model_name}': {e}", file=sys.stderr)
            continue

        available += 1
        with_tips_len = _count_prompt_tokens(
            prompt_with_tips,
            tok,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
        )
        if with_tips_len > max_prompt_tokens:
            return True

        # Also sanity-check that removing tips doesn't exceed the limit. If it does,
        # dropping tips won't help, but we still drop since requested behavior is
        # "if tips makes it exceed, remove tips".
        _ = _count_prompt_tokens(
            prompt_without_tips,
            tok,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
        )

    if available == 0:
        print(
            "[length-guard] warning: no tokenizer loaded; skip prompt length guard",
            file=sys.stderr,
        )
    return False


def extract_instruction_from_bigcodebench(complete_prompt: str) -> str:
    """Extract the instruction/docstring from BigCodeBench complete_prompt."""
    # BigCodeBench format: function signature with docstring
    # Extract the docstring content between triple quotes
    if '"""' in complete_prompt:
        parts = complete_prompt.split('"""')
        if len(parts) >= 2:
            return parts[1].strip()
    elif "'''" in complete_prompt:
        parts = complete_prompt.split("'''")
        if len(parts) >= 2:
            return parts[1].strip()

    # Fallback: try to extract from the beginning of the function
    lines = complete_prompt.strip().split('\n')
    instruction_lines = []
    in_docstring = False

    for line in lines:
        if '"""' in line:
            if not in_docstring:
                in_docstring = True
                # Get content after the opening """
                content = line.split('"""', 1)[1]
                if content.strip():
                    instruction_lines.append(content)
            else:
                # Closing """
                break
        elif in_docstring:
            instruction_lines.append(line)

    return '\n'.join(instruction_lines).strip()


def extract_instruction_from_instruct_prompt(instruct_prompt: str) -> str:
    """Extract the core instruction from BigCodeBench instruct_prompt."""
    # instruct_prompt format: description + args + output + code template
    # We want to extract just the description part before "The function should output with:"

    lines = instruct_prompt.strip().split('\n')
    instruction_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith('The function should output with:'):
            break
        if line.startswith('You should write self-contained code'):
            break
        if line.startswith('```'):
            break
        if line and not line.startswith('Args:'):
            instruction_lines.append(line)

    return '\n'.join(instruction_lines).strip()


def generate_tips_batch(instructions: list, client: OpenAI, tips_prompt: str = None, max_workers: int = 8) -> list:
    """Generate tips for multiple instructions using the vLLM endpoint with Completion API."""
    
    # 1. 这里的 Template 必须和您的 SFT 脚本里的 --input_template 一字不差！
    # 直接使用您的 SFT 训练模板格式
    
    template = (
        "<|im_start|>system\n"
        "You are an expert programming mentor. Analyze the following coding instruction first, followed by the separator \"###TIPS###\", and then provide specific tips to help developers implement this correctly.\n"
        "Do not provide solution or any solution-related ideas. Your only task is to provide tips to clarify the instruction and identify potential pitfalls, common mistakes, and key considerations during solving the problem.\n"
        "Your tips should be problem-specific, not general programming principles. Do not conclude debug/test/verify tips.<|im_end|>\n"
        "<|im_start|>user\n"
        "Instruction: {}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )



    print(f"DEBUG: Template:\n{template}\n" + "-"*20)
    # 2. 构造纯文本 Prompts - 与 SFT 训练完全一致
    raw_prompts = [template.format(ins) for ins in instructions]

    import concurrent.futures

    def single_tip_generation(prompt):
        try:
            # 获取可用的模型列表，使用第一个模型
            model_name = client.models.list().data[0].id

            # 3. 关键修改：使用 completions API，传入完整的 prompt
            response = client.completions.create(
                model=model_name,
                prompt=prompt,       # 传入纯字符串
                temperature=0.0,     # 锁死随机性
                max_tokens=4096,
                # 加上停止符，防止模型停不下来
                stop=["<|endoftext|>", "<|im_end|>", "<|im_start|>"]
            )
            raw_output = response.choices[0].text.strip()
            
            # 处理输出，提取 ###TIPS### 之后的内容
            tips = raw_output
            if "###TIPS###" in raw_output:
                tips = raw_output.split("###TIPS###")[-1].strip()
            else:
                # 如果没有找到分隔符，假设整个输出都是tips（或者根据情况处理）
                # 但根据用户的要求，应该是有分隔符的。如果没有，可能是模型没有遵循指令，或者输出了NO_TIPS
                if "###NO_TIPS###" in raw_output:
                    return ""
                tips = raw_output.strip()

            if 'debugging conversation:' in tips:
                tips = tips.split('debugging conversation:')[-1].strip()
            

            return tips
        except Exception as e:
            print(f"Error: {e}")
            return ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(single_tip_generation, raw_prompts))

    return results


def combine_instruction_with_tips(original_instruction: str, tips: str) -> str:
    """Combine original instruction with generated tips."""
    if not tips or tips.strip() == "" or tips.lower().strip() == "none" or tips.strip() == "None" or "###NO_ACTION" in tips or "no_tip" in tips.lower():
        return original_instruction

    # 直接使用原始tips，不进行格式化
    enhanced_instruction = f"{original_instruction}\n{tips.strip()}"
    return enhanced_instruction


def enhance_instruct_prompt(original_instruct_prompt: str, tips: str, task_id: str) -> str:
    """Enhance instruct_prompt by adding tips in appropriate location."""
    
    tips = "#no_tip" if "no_tip" in tips.lower() or "no_action" in tips.lower() else tips
    print(f"Task {task_id}: {tips}", file=sys.stderr)
    if not tips or tips.strip() == "" or tips.lower().strip() == "none" or tips.strip() == "None" or "#no_tip" in tips.lower() or "no_action" in tips.lower():
        return original_instruct_prompt

    lines = original_instruct_prompt.strip().split('\n')
    enhanced_lines = []

    # Find where to insert tips (after the main description but before technical details)
    inserted_tips = False

    for line in lines:
        # Insert tips after the main description but before "The function should output with:"
        if not inserted_tips and line.strip().startswith('The function should output with:'):
            # Insert tips before the output specification
            # Add an empty line before tips for better separation
            if enhanced_lines and enhanced_lines[-1].strip():
                enhanced_lines.append("")
            enhanced_lines.append(f"Tips:\n{tips.strip()}\n")
            inserted_tips = True
        
        enhanced_lines.append(line)

    # If we didn't find a good place to insert, append at the end
    if not inserted_tips:
        if enhanced_lines and enhanced_lines[-1].strip():
            enhanced_lines.append("")
        enhanced_lines.append(f"Tips:\n{tips.strip()}")

    return '\n'.join(enhanced_lines)



def load_and_cache_bigcodebench_dataset(local_cache_dir: str = "./datasets/bigcodebench"):
    """Load BigCodeBench dataset and cache it locally."""
    import json

    # Ensure cache directory exists
    os.makedirs(local_cache_dir, exist_ok=True)

    # Check if we have a cached version locally
    cache_file = os.path.join(local_cache_dir, "bigcodebench_original.jsonl")

    if os.path.exists(cache_file):
        print(f"✓ Loading BigCodeBench dataset from local cache: {cache_file}")
        dataset = {}
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                task = json.loads(line.strip())
                dataset[task["task_id"]] = task
    else:
        print("Downloading BigCodeBench dataset from remote repository...")
        try:
            from bigcodebench.data import get_bigcodebench
            dataset = get_bigcodebench()
            print(f"✓ Downloaded BigCodeBench dataset with {len(dataset)} samples")

            # Save to local cache
            print(f"Saving dataset to local cache: {cache_file}")
            with open(cache_file, 'w', encoding='utf-8') as f:
                for task in dataset.values():
                    json.dump(task, f, ensure_ascii=False)
                    f.write('\n')

        except Exception as e:
            print(f"Error loading BigCodeBench dataset: {e}")
            sys.exit(1)

    return dataset


def rewrite_bigcodebench_dataset(
    output_path: str,
    api_base: str,
    mode: str = "tips",
    batch_size: int = 8,
    max_workers: int = 8,
    local_cache_dir: str = "./datasets/bigcodebench",
    sample_only: int = None,
    max_prompt_tokens: int = 2048,
    tokenizer_models: list[str] | None = None,
    trust_remote_code: bool = False,
):
    """Rewrite instructions in the BigCodeBench dataset and save to output_path."""
    # Initialize OpenAI client for vLLM
    client = OpenAI(
        api_key="EMPTY",
        base_url=api_base,
    )

    # Load BigCodeBench dataset from local cache or download and cache it
    dataset = load_and_cache_bigcodebench_dataset(local_cache_dir)
    print(f"Loaded BigCodeBench dataset with {len(dataset)} samples")

    # Ensure output directory exists (use local cache dir if output_path is relative)
    if not os.path.isabs(output_path):
        output_path = os.path.join(local_cache_dir, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Process dataset
    processed_samples = []
    tasks = list(dataset.items())

    if sample_only:
        tasks = tasks[:sample_only]
        print(f"Processing only first {sample_only} samples for testing")

    print(f"Using batch_size={batch_size}, max_workers={max_workers}, mode={mode}")

    if not tokenizer_models:
        # Default to the two models used in evalplus_exe.sh in this repo.
        tokenizer_models = _parse_csv(os.getenv("REWRITE_TOKENIZER_MODELS"))
        if not tokenizer_models:
            tokenizer_models = [
                os.environ["QWEN_CODER_MODEL"],
                os.environ["DS_CODER_MODEL"],
            ]

    # 分批处理数据集
    for i in tqdm(range(0, len(tasks), batch_size), desc="Processing batches"):
        batch_tasks = tasks[i:i+batch_size]
        batch_instructions = []
        batch_complete_prompts = []
        batch_instruct_prompts = []

        # 提取当前批次的instructions
        for task_id, task in batch_tasks:
            complete_prompt = task["complete_prompt"]
            instruct_prompt = task["instruct_prompt"]

            # 提取两种prompt中的指令内容
            complete_instruction = extract_instruction_from_bigcodebench(complete_prompt)
            instruct_instruction = extract_instruction_from_instruct_prompt(instruct_prompt)

            batch_instructions.append((complete_instruction, instruct_instruction))
            batch_complete_prompts.append(complete_prompt)
            batch_instruct_prompts.append(instruct_prompt)

        # 根据模式处理instructions
        if mode == "tips":
            # Tips模式：分别为complete和instruct指令生成tips
            complete_instructions = [instr[0] for instr in batch_instructions]
            instruct_instructions = [instr[1] for instr in batch_instructions]

            # Use the same full prompt surface that the coder sees.  The
            # extracted instruct_instruction stops before the output contract
            # and code template, which is especially harmful for BCB tasks.
            complete_tips_list = generate_tips_batch(
                batch_complete_prompts, client, None, max_workers
            )
            instruct_tips_list = generate_tips_batch(
                batch_instruct_prompts, client, None, max_workers
            )

            # 合并tips: (complete_tips, instruct_tips)
            tips_list = list(zip(complete_tips_list, instruct_tips_list))
        else:
            # 如果将来支持其他模式
            tips_list = [("", "")] * len(batch_instructions)

        # 处理当前批次的结果
        for j, (task_id, task) in enumerate(batch_tasks):
            complete_prompt = batch_complete_prompts[j]
            instruct_prompt = batch_instruct_prompts[j]
            complete_instruction, instruct_instruction = batch_instructions[j]

            if mode == "tips":
                # Tips模式：分别改写两种prompt
                complete_tips, instruct_tips = tips_list[j] if j < len(tips_list) else ("", "")

                # 改写complete_prompt
                enhanced_prompt = complete_prompt
                if complete_tips and complete_tips.strip() and complete_tips.strip() != "None":
                    enhanced_prompt = complete_prompt.rstrip()
                    if '"""' in enhanced_prompt:
                        # Find last """
                        last_quote_idx = enhanced_prompt.rfind('"""')
                        enhanced_prompt = enhanced_prompt[:last_quote_idx] + '\n' + complete_tips.strip() + '\n' + enhanced_prompt[last_quote_idx:]
                    else:
                        enhanced_prompt = complete_prompt
                    if not enhanced_prompt.endswith('\n'):
                        enhanced_prompt += '\n'

                # Length guard for complete_prompt
                if enhanced_prompt != complete_prompt and max_prompt_tokens is not None:
                    if _should_drop_tips(
                        prompt_with_tips=enhanced_prompt,
                        prompt_without_tips=complete_prompt,
                        tokenizer_models=tokenizer_models,
                        max_prompt_tokens=max_prompt_tokens,
                        trust_remote_code=trust_remote_code,
                        instruction_prefix=None,
                        response_prefix=None,
                    ):
                        print(
                            f"[length-guard] drop tips for task {task_id} (complete_prompt): exceeds {max_prompt_tokens}",
                            file=sys.stderr,
                        )
                        enhanced_prompt = complete_prompt
                        complete_tips = "" # Clear tips if dropped

                # 改写instruct_prompt
                enhanced_instruct_prompt = enhance_instruct_prompt(instruct_prompt, instruct_tips, task_id)

                # Length guard for instruct_prompt
                if enhanced_instruct_prompt != instruct_prompt and max_prompt_tokens is not None:
                    if _should_drop_tips(
                        prompt_with_tips=enhanced_instruct_prompt,
                        prompt_without_tips=instruct_prompt,
                        tokenizer_models=tokenizer_models,
                        max_prompt_tokens=max_prompt_tokens,
                        trust_remote_code=trust_remote_code,
                        instruction_prefix=None,
                        response_prefix=None,
                    ):
                        print(
                            f"[length-guard] drop tips for task {task_id} (instruct_prompt): exceeds {max_prompt_tokens}",
                            file=sys.stderr,
                        )
                        enhanced_instruct_prompt = instruct_prompt
                        instruct_tips = "" # Clear tips if dropped

            else:
                enhanced_prompt = complete_prompt
                enhanced_instruct_prompt = instruct_prompt
                complete_tips, instruct_tips = "", ""

            # Create modified task
            modified_task = task.copy()
            modified_task["complete_prompt"] = enhanced_prompt
            modified_task["instruct_prompt"] = enhanced_instruct_prompt
            modified_task["original_complete_instruction"] = complete_instruction
            modified_task["original_instruct_instruction"] = instruct_instruction
            modified_task["enhanced_complete_instruction"] = combine_instruction_with_tips(complete_instruction, complete_tips)
            modified_task["enhanced_instruct_instruction"] = combine_instruction_with_tips(instruct_instruction, instruct_tips)
            if mode == "tips":
                modified_task["generated_complete_tips"] = complete_tips
                modified_task["generated_instruct_tips"] = instruct_tips

            processed_samples.append(modified_task)

    # Save to JSONL
    print(f"Saving rewritten dataset to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for task in processed_samples:
            json.dump(task, f, ensure_ascii=False)
            f.write('\n')

    print(f"✓ Successfully saved {len(processed_samples)} tasks")
    print(f"\nTo use with bigcodebench, set environment variable:")
    print(f'export BIGCODEBENCH_OVERRIDE_PATH="{output_path}"')

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Rewrite BigCodeBench dataset instructions using vLLM")
    parser.add_argument("--output", type=str, default="bigcodebench_with_tips.jsonl",
                        help="Output path for rewritten dataset (default: bigcodebench_with_tips.jsonl)")
    parser.add_argument("--api-base", type=str, default="http://localhost:30000/v1",
                        help="vLLM endpoint URL (default: http://localhost:30000/v1)")
    parser.add_argument("--mode", type=str, choices=["tips"], default="tips",
                        help="Mode: 'tips' for adding tips to original instruction")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for processing (default: 8)")
    parser.add_argument("--max-workers", type=int, default=16,
                        help="Maximum number of concurrent workers (default: 16)")
    parser.add_argument("--local-cache-dir", type=str, default="./datasets/bigcodebench",
                        help="Local directory to cache datasets (default: ./datasets/bigcodebench)")
    parser.add_argument("--sample-only", type=int, default=None,
                        help="Only process first N samples (for testing)")
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=int(os.getenv("REWRITE_MAX_PROMPT_TOKENS", "2048")),
        help="If mode=tips and prompt+tips exceeds this token length (per tokenizer), drop tips entirely (default: 2048)",
    )
    parser.add_argument(
        "--tokenizer-models",
        type=str,
        default=os.getenv("REWRITE_TOKENIZER_MODELS", ""),
        help="Comma-separated HF model names/paths used to estimate token length; if any exceeds, tips are dropped",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=bool(int(os.getenv("REWRITE_TRUST_REMOTE_CODE", "0"))),
        help="Pass trust_remote_code=True to AutoTokenizer (default: false)",
    )

    args = parser.parse_args()

    rewrite_bigcodebench_dataset(
        output_path=args.output,
        api_base=args.api_base,
        mode=args.mode,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        local_cache_dir=args.local_cache_dir,
        sample_only=args.sample_only,
        max_prompt_tokens=args.max_prompt_tokens,
        tokenizer_models=_parse_csv(args.tokenizer_models),
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
