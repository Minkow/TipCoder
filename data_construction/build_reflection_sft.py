"""
构建 Reflection-based SFT 数据集

工作流:
1. Easy 样本: output = <eos>
2. Hard 样本:
   - 执行代码获取 feedback (失败的测试用例 + 报错)
   - 通过 OpenAI API 迭代改进代码
   - 保存完整 reflection chain
   - 基于 reflection 生成改进的 instruction
"""

import json
import fire
import os
import time
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI

# Add parent directory to path
from openrlhf.cli.eval_test_cases import evaluate
from evalplus.sanitize import code_extract


def load_json_file(path: str) -> Any:
    """Helper to load JSON with error handling."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"  Error: JSON file {path} is corrupted: {e}")
        print(
            "  This file will be ignored. Consider deleting it if the error persists."
        )
        return None
    except Exception as e:
        print(f"  Error: Failed to load JSON file {path}: {e}")
        return None


def save_json_file(path: str, data: Any) -> None:
    """Helper to write JSON with UTF-8 encoding."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()  # 确保数据立即写入磁盘


def select_initial_codes(
    sample: Dict[str, Any], max_codes: int = 2
) -> List[Dict[str, Any]]:
    """
    选择初始代码进行 reflection，从不同分位数分层选取

    Args:
        sample: 样本数据
        max_codes: 最多选择的代码数量

    Returns:
        选中的代码列表，包含 completion 和 completion_id
    """
    inferences = sample.get("inferences", [])
    qwen_inferences = [
        inf for inf in inferences if inf.get("model_name") == "qwen_coder_2.5"
    ]

    print(f"    Found {len(qwen_inferences)} qwen_coder_2.5 inferences")

    # 选择未全部通过的代码（pass_rate < 1.0）
    incomplete_codes = [
        {
            "completion": inf.get("completion", ""),
            "completion_id": inf.get("completion_id", -1),
            "pass_rate": inf.get("pass_rate", 0.0),
        }
        for inf in qwen_inferences
        if inf.get("pass_rate", 0.0) < 1.0
    ]

    print(f"    {len(incomplete_codes)} codes with pass_rate < 1.0")

    if len(incomplete_codes) == 0:
        return []

    # 按 pass_rate 升序排序
    incomplete_codes.sort(key=lambda x: x["pass_rate"])

    # 分层选取：从不同分位数选择代码，避免扎堆
    selected = []
    if max_codes == 1:
        # 只选 1 个：选中位数
        idx = len(incomplete_codes) // 2
        selected = [incomplete_codes[idx]]
    elif max_codes == 2:
        # 选 2 个：1/3 和 2/3 分位
        idx1 = len(incomplete_codes) // 3
        idx2 = len(incomplete_codes) * 2 // 3
        selected = [incomplete_codes[idx1], incomplete_codes[idx2]]
    elif max_codes >= 3 and len(incomplete_codes) >= 3:
        # # 选 3+ 个：均匀分布
        # indices = [int(len(incomplete_codes) * i / max_codes) for i in range(max_codes)]
        # selected = [incomplete_codes[idx] for idx in indices if idx < len(incomplete_codes)]
        idx1 = len(incomplete_codes) // 3
        idx2 = len(incomplete_codes) * 2 // 3
        selected = [
            incomplete_codes[idx1],
            incomplete_codes[idx2],
            incomplete_codes[-1],
        ]
    else:
        selected = incomplete_codes

    if len(selected) > 0:
        pass_rates_str = [f"{c['pass_rate']:.2%}" for c in selected]
        print(
            f"    Selected {len(selected)} codes (stratified) with pass_rates: {pass_rates_str}"
        )
    else:
        print(f"    WARNING: No incomplete codes found (all pass_rate >= 1.0)")

    return selected


def execute_and_get_feedback(
    code: str, task_id: str, test_cases: List[str], n_workers: int = 8
) -> Dict[str, Any]:
    """
    执行代码并获取详细的 feedback

    Returns:
        {
            'pass_rate': float,
            'passed_tests': List[bool],
            'failed_indices': List[int],
            'total_tests': int,
            'solution': str  # 实际执行的代码
        }
    """
    eval_sample = {
        "task_id": task_id,
        "output": code,
        "tests": test_cases,
        "_identifier": f"{task_id}_feedback",
    }

    # 评估代码
    eval_results, pass_rates = evaluate(
        samples=[eval_sample],
        dataset=None,
        base_only=True,
        n_workers=n_workers,
        test_details=True,
        extract_solution=True,
        output_file=None,
    )

    result = eval_results[0]
    passed_tests = result["base"][1]  # List[bool]
    failed_indices = [i for i, passed in enumerate(passed_tests) if not passed]

    return {
        "pass_rate": result["pass_rate"],
        "passed_tests": passed_tests,
        "failed_indices": failed_indices,
        "total_tests": len(passed_tests),
        "solution": result["solution"],
    }


def generate_feedback_prompt(
    instruction: str,
    code: str,
    test_cases: List[str],
    failed_indices: List[int],
) -> str:
    """
    生成发送给 OpenAI 的反馈提示
    """
    failed_tests = [test_cases[i] for i in failed_indices]

    prompt = f"""You are given a coding problem and a solution that failed some test cases.

**Problem:**
{instruction}

**Your Previous Solution:**
```python
{code}
```

**Failed Test Cases:**
"""
    for i, test in enumerate(failed_tests, 1):
        prompt += f"{i}. {test}\n"

    prompt += """
**Task:**
Please fix the code to pass all test cases. 
"""

    return prompt


def call_openai_api(
    prompt: str,
    api_key: str,
    api_base: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    system_message: str = "You are a helpful coding assistant.",
    max_retries: int = 1,
    retry_delay: float = 10.0,
) -> str:
    """
    调用 OpenAI API 生成改进的代码（带重试机制）

    Args:
        prompt: 用户提示
        api_key: API 密钥
        api_base: API base URL
        model: 模型名称
        temperature: 温度参数
        system_message: 系统消息
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）

    Returns:
        生成的文本
    """
    # 创建客户端

    client = OpenAI(api_key=api_key, base_url=api_base, timeout=180)

    last_error = None

    for attempt in range(max_retries):
        try:
            print("  >> Sending request to OpenAI API...")
            # print(f"     model: {model}, temperature: {temperature}")
            # print("----- LLM Request Start -----")
            # print(prompt)
            # print("----- LLM Request End -----")

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=32768,
                temperature=temperature,
            )

            reply_content = response.choices[0].message.content.strip()

            print("----- LLM Response Start -----")
            print(reply_content)
            print("----- LLM Response End -----")

            return reply_content

        except Exception as e:
            last_error = e
            error_msg = str(e)

            print(
                f"  Error: {error_msg[:100]}, retrying (attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(retry_delay * (attempt + 1))

    # 所有重试都失败
    raise Exception(
        f"API call failed after {max_retries} attempts. Last error: {last_error}"
    )


def reflection_loop(
    instruction: str,
    initial_code: str,
    test_cases: List[str],
    task_id: str,
    api_key: str,
    api_base: str,
    max_steps: int = 4,
    n_workers: int = 8,
    openai_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    执行 reflection loop，保存完整的对话历史

    Returns:
        {
            'success': bool,
            'reflection_chain': List[{
                'step': int,
                'code': str,
                'feedback': {...},
                'prompt': str (发送给模型的提示),
                'model_reply': str (模型的完整回复),
                'dialogue': [{role, content}, ...] (完整对话历史)
            }],
            'final_pass_rate': float
        }
    """
    reflection_chain = []
    current_code = initial_code
    dialogue_history = []  # 累积的对话历史
    current_feedback = None  # 保存当前反馈，用于下一轮

    for step in range(max_steps):
        # 执行代码获取反馈
        feedback = execute_and_get_feedback(
            code=current_code,
            task_id=task_id,
            test_cases=test_cases,
            n_workers=n_workers,
        )

        chain_entry = {
            "step": step,
            "code": current_code,
            "feedback": feedback,
            "dialogue": list(dialogue_history),  # 保存当前步骤的完整对话历史
        }

        # 检查是否成功
        if feedback["pass_rate"] == 1.0:
            chain_entry["success_at_this_step"] = True
            reflection_chain.append(chain_entry)
            return {
                "success": True,
                "reflection_chain": reflection_chain,
                "final_pass_rate": 1.0,
                "total_steps": step + 1,
            }

        chain_entry["success_at_this_step"] = False

        # 如果不是最后一步，生成改进提示并调用 API
        if step < max_steps - 1:
            # 使用当前这一轮的失败测试用例
            prompt = generate_feedback_prompt(
                instruction=instruction,
                code=current_code,
                test_cases=test_cases,
                failed_indices=feedback["failed_indices"],
            )
            chain_entry["prompt"] = prompt

            # 调用 OpenAI API
            try:
                model_reply = call_openai_api(
                    prompt=prompt,
                    api_key=api_key,
                    api_base=api_base,
                    model=openai_model,
                )
                chain_entry["model_reply"] = model_reply

                # 更新对话历史
                dialogue_history.append({"role": "user", "content": prompt})
                dialogue_history.append({"role": "assistant", "content": model_reply})

                # 从回复中提取代码（使用 evalplus 的成熟实现）
                try:
                    extracted_code = code_extract(
                        model_reply.encode("utf-8", "ignore")
                        .decode("utf-8")
                        .replace("\x00", "")
                    )
                    # 如果提取成功且不为空，使用提取的代码；否则使用原始回复
                    if extracted_code and extracted_code.strip():
                        current_code = extracted_code
                    else:
                        current_code = model_reply
                except Exception as e:
                    print(f"  Code extraction failed: {e}, using raw reply")
                    current_code = model_reply

            except Exception as e:
                print(f"  OpenAI API call failed: {e}")
                chain_entry["api_error"] = str(e)
                chain_entry["model_reply"] = None
                reflection_chain.append(chain_entry)
                break

        reflection_chain.append(chain_entry)

    # 最后一步仍未成功
    return {
        "success": False,
        "reflection_chain": reflection_chain,
        "final_pass_rate": reflection_chain[-1]["feedback"]["pass_rate"]
        if reflection_chain
        else 0.0,
        "total_steps": len(reflection_chain),
    }


def generate_improved_instruction(
    original_instruction: str,
    reflection_data: Dict[str, Any],
    api_key: str,
    api_base: str,
    openai_model: str = "gpt-4o-mini",
) -> str:
    """
    基于 reflection data 生成改进的 instruction
    """
    reflection_chain = reflection_data["reflection_chain"]

    # 构建 reflection 摘要
    reflection_summary = f"Original Instruction:\n{original_instruction}\n\n"
    reflection_summary += "Reflection Process:\n"

    for entry in reflection_chain:
        step = entry["step"]
        pass_rate = entry["feedback"]["pass_rate"]
        failed_count = len(entry["feedback"]["failed_indices"])
        reflection_summary += f"Step {step + 1}: Pass Rate = {pass_rate:.2%}, Failed {failed_count} tests\n"

    if reflection_data["success"]:
        reflection_summary += "\nFinal Status: Successfully passed all tests\n"
    else:
        reflection_summary += f"\nFinal Status: Did not pass all tests (final pass rate: {reflection_data['final_pass_rate']:.2%})\n"

    # 生成改进提示
    prompt = f"""You are tasked with improving a coding problem instruction based on a reflection process.

{reflection_summary}

**Task:**
Based on the reflection process above, analyze what aspects of the original instruction might be unclear, ambiguous, or misleading. 
Then provide an IMPROVED version of the instruction that:
1. Clarifies any ambiguous points
2. Adds important constraints or edge cases that were not obvious
3. Makes the problem statement more precise and complete

Provide ONLY the improved instruction text without any meta-commentary.
"""

    try:
        improved_instruction = call_openai_api(
            prompt=prompt,
            api_key=api_key,
            api_base=api_base,
            model=openai_model,
            temperature=0.3,  # 更低的温度以获得更稳定的输出
        )
        return improved_instruction
    except Exception as e:
        print(f"Failed to generate improved instruction: {e}")
        return original_instruction  # 失败时返回原始 instruction


def process_hard_sample(
    sample: Dict[str, Any],
    api_key: str,
    api_base: str,
    max_codes: int = 2,
    max_steps: int = 4,
    n_workers: int = 8,
    openai_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    处理单个 hard 样本
    """
    task_id = sample.get("id", "unknown")
    instruction = sample.get("question", "")
    test_cases = sample.get("test_cases", [])

    # 选择初始代码
    initial_codes = select_initial_codes(sample, max_codes=max_codes)

    if len(initial_codes) == 0:
        # 没有未通过的代码，可能所有代码都通过了
        print(
            f"  No incomplete codes, skipping reflection (all codes passed or no qwen_coder_2.5 codes)"
        )
        return {
            "id": task_id,
            "source": sample.get("source"),
            "difficulty": "hard",
            "prompt": instruction,
            "output": "<eos>",  # 所有代码都通过，无需改进
            "reflection_data": {
                "reason": "all_codes_passed_or_no_codes",
                "all_reflections": [],
                "best_reflection": None,
                "best_pass_rate": 1.0,
            },
        }

    # 对每个初始代码执行 reflection loop (retry 机制)
    # 遍历所有选中的代码，直到找到一个能够全部通过的，或者尝试完所有代码
    all_reflections = []
    best_reflection = None
    best_pass_rate = -1
    success_found = False

    for idx, code_data in enumerate(initial_codes):
        print(
            f"  Running reflection for code {idx + 1}/{len(initial_codes)}, "
            f"completion_id={code_data['completion_id']} (initial pass_rate: {code_data['pass_rate']:.2%})"
        )

        reflection_result = reflection_loop(
            instruction=instruction,
            initial_code=code_data["completion"],
            test_cases=test_cases,
            task_id=task_id,
            api_key=api_key,
            api_base=api_base,
            max_steps=max_steps,
            n_workers=n_workers,
            openai_model=openai_model,
        )

        reflection_result["initial_completion_id"] = code_data["completion_id"]
        reflection_result["initial_pass_rate"] = code_data["pass_rate"]
        all_reflections.append(reflection_result)

        # 跟踪最佳 reflection
        if reflection_result["final_pass_rate"] > best_pass_rate:
            best_pass_rate = reflection_result["final_pass_rate"]
            best_reflection = reflection_result

        # 如果成功（all pass），停止尝试其他代码
        if reflection_result["success"]:
            print(
                f"  ✓ Success! Achieved 100% pass rate in {reflection_result.get('total_steps', 0)} steps"
            )
            success_found = True
            break
        else:
            print(
                f"  ✗ Failed to achieve 100% pass rate (best: {reflection_result['final_pass_rate']:.2%})"
            )
            # 继续尝试下一个初始代码

    # 只保存成功的案例（pass_rate = 1.0）
    if not success_found or best_pass_rate < 1.0:
        print(
            f"  ✗ All {len(initial_codes)} codes failed to achieve 100% pass rate. Best: {best_pass_rate:.2%}. Skipping this sample."
        )
        return None  # 返回 None 表示失败，调用方需要过滤掉

    # TODO: 基于最佳 reflection 生成改进的 instruction (后续实现更复杂的工作流)
    # print(f"  Generating improved instruction (best pass_rate: {best_pass_rate:.2%})")
    # improved_instruction = generate_improved_instruction(
    #     original_instruction=instruction,
    #     reflection_data=best_reflection,
    #     api_key=api_key,
    #     api_base=api_base,
    #     openai_model=openai_model
    # )

    print(
        f"  Best reflection achieved {best_pass_rate:.2%} pass rate in {best_reflection.get('total_steps', 0)} steps"
    )

    return {
        "id": task_id,
        "source": sample.get("source"),
        "difficulty": "hard",
        "prompt": instruction,
        "output": None,  # 暂时不生成，等待后续更复杂的工作流
        "reflection_data": {
            "all_reflections": all_reflections,
            "best_reflection": best_reflection,
            "best_pass_rate": best_pass_rate,
        },
    }


def main(
    hard_input_path: str = "./data/acecode_87K/acecode_87k_hard_sft_filtered.json",
    output_path: str = "./data/acecode_87K/acecode_reflection_sft.json",
    reflection_data_path: str = "./data/acecode_87K/acecode_reflection_data.json",
    api_key: str = None,
    api_base: str = "https://api.openai.com/v1",
    n_samples: int = 0,
    max_codes_per_sample: int = 3,
    max_reflection_steps: int = 3,
    n_workers: int = 8,
    openai_model: str = "gemini-3-pro-all",
):
    """
    构建 Reflection-based SFT 数据集

    Args:
        hard_input_path: hard 样本输入路径
        output_path: 输出 SFT 数据集路径
        reflection_data_path: 保存完整 reflection 数据的路径
        api_key: OpenAI API key
        api_base: OpenAI API base URL
        n_samples: 处理的样本数量（0 表示全部）
        max_codes_per_sample: 每个样本最多处理的代码数量
        max_reflection_steps: reflection loop 的最大步数
        n_workers: 评估时的并行 worker 数
        openai_model: OpenAI 模型名称
    """
    # 获取 API key
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Please provide api_key or set OPENAI_API_KEY environment variable"
            )

    print("=" * 60)
    print("Reflection-based SFT Dataset Builder")
    print("=" * 60)
    print(f"Hard input: {hard_input_path}")
    print(f"Output: {output_path}")
    print(f"Reflection data: {reflection_data_path}")
    print(f"OpenAI model: {openai_model}")
    print(f"Max codes per sample: {max_codes_per_sample}")
    print(f"Max reflection steps: {max_reflection_steps}")
    print("=" * 60)

    # 加载数据
    print("\nLoading data...")
    with open(hard_input_path, "r") as f:
        try:
            hard_samples = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            hard_samples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(hard_samples)} hard samples")

    # 检查是否有已存在的结果文件（重试机制）
    existing_solved_ids = set()
    existing_sft_dataset = []

    # 优先检查partial文件，然后检查主文件
    files_to_check = [reflection_data_path + ".partial.json", reflection_data_path]
    loaded_from_file = None

    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"\nFound existing reflection data at {file_path}")
            print("Loading existing results to identify solved cases...")
            loaded_data = load_json_file(file_path)

            if loaded_data is None:
                print(f"  Warning: Skipping corrupted file {file_path}")
                continue

            existing_sft_dataset = loaded_data

            # 找出已解决的case IDs
            for entry in existing_sft_dataset:
                entry_id = entry.get("id")
                if entry_id is None:
                    continue

                # 判断是否已解决：检查reflection_data中的best_reflection是否success=True
                reflection_data = entry.get("reflection_data")
                if reflection_data:
                    # 方法1: 检查best_reflection的success字段
                    best_reflection = reflection_data.get("best_reflection")
                    if best_reflection and best_reflection.get("success", False):
                        existing_solved_ids.add(entry_id)
                    # 方法2: 检查best_pass_rate是否为1.0
                    elif reflection_data.get("best_pass_rate", 0.0) == 1.0:
                        existing_solved_ids.add(entry_id)
                    # 方法3: 检查reason是否为all_codes_passed_or_no_codes（这种情况也视为已解决）
                    elif (
                        reflection_data.get("reason") == "all_codes_passed_or_no_codes"
                    ):
                        existing_solved_ids.add(entry_id)

            print(f"  Found {len(existing_solved_ids)} already solved cases")
            print(f"  Existing dataset contains {len(existing_sft_dataset)} entries")
            loaded_from_file = file_path
            break  # 找到第一个有效的文件就停止

    if loaded_from_file:
        print(f"Successfully loaded data from {loaded_from_file}")
    else:
        print(f"\nNo existing reflection data found at {files_to_check}")
        print("Will process all samples from scratch")

    # 过滤出未解决的hard样本
    hard_sample_ids = {sample.get("id") for sample in hard_samples}
    solved_in_current_hard = (
        existing_solved_ids & hard_sample_ids
    )  # 只统计存在于当前hard样本中的已解决样本

    unsolved_hard_samples = [
        sample for sample in hard_samples if sample.get("id") not in existing_solved_ids
    ]

    print(f"\nFiltering unsolved cases:")
    print(f"  Total hard samples: {len(hard_samples)}")
    print(f"  Solved in reflection data: {len(existing_solved_ids)}")
    print(f"  Solved in current hard samples: {len(solved_in_current_hard)}")
    print(f"  Remaining unsolved: {len(unsolved_hard_samples)}")
    if len(existing_solved_ids) > len(solved_in_current_hard):
        print(
            f"  Note: {len(existing_solved_ids) - len(solved_in_current_hard)} solved cases not in current hard dataset"
        )

    # 限制样本数量用于测试（只限制未解决的样本）
    if n_samples > 0:
        unsolved_hard_samples = unsolved_hard_samples[:n_samples]
        print(
            f"  Processing {len(unsolved_hard_samples)} unsolved samples for testing (n_samples={n_samples})"
        )

    # 初始化sft_dataset：先加载已有的已解决样本
    sft_dataset = existing_sft_dataset.copy() if existing_sft_dataset else []
    existing_ids = {
        entry.get("id") for entry in sft_dataset if entry.get("id") is not None
    }

    # 尝试加载已有的sft输出文件（优先partial文件）
    sft_files_to_check = [output_path + ".partial.json", output_path]
    for sft_file_path in sft_files_to_check:
        if os.path.exists(sft_file_path):
            print(f"\nFound existing SFT data at {sft_file_path}")
            print("Loading existing SFT results...")
            loaded_sft_data = load_json_file(sft_file_path)

            if loaded_sft_data is not None:
                sft_dataset = loaded_sft_data
                existing_ids = {
                    entry.get("id")
                    for entry in sft_dataset
                    if entry.get("id") is not None
                }
                print(f"  Loaded {len(sft_dataset)} existing SFT entries")
                print(f"Successfully loaded SFT data from {sft_file_path}")
                break
            else:
                print(f"  Warning: Skipping corrupted SFT file {sft_file_path}")
                continue

    incremental_output_path = output_path + ".partial.json"
    incremental_reflection_path = reflection_data_path + ".partial.json"

    # 处理 hard 样本（只处理未解决的）
    if len(unsolved_hard_samples) > 0:
        print(
            f"\nProcessing {len(unsolved_hard_samples)} unsolved hard samples with reflection..."
        )
        new_successful_count = 0
        new_failed_count = 0

        for idx, sample in enumerate(
            tqdm(unsolved_hard_samples, desc="Processing unsolved hard samples")
        ):
            print(
                f"\n[{idx + 1}/{len(unsolved_hard_samples)}] Processing sample {sample.get('id')}"
            )

            sft_entry = process_hard_sample(
                sample=sample,
                api_key=api_key,
                api_base=api_base,
                max_codes=max_codes_per_sample,
                max_steps=max_reflection_steps,
                n_workers=n_workers,
                openai_model=openai_model,
            )

            # 只保存成功的案例（pass_rate = 1.0）
            if sft_entry is not None:
                entry_id = sft_entry.get("id")
                replaced_existing = False

                if entry_id is not None and entry_id in existing_ids:
                    for existing_index, existing_entry in enumerate(sft_dataset):
                        if existing_entry.get("id") == entry_id:
                            sft_dataset[existing_index] = sft_entry
                            replaced_existing = True
                            print(f"  ↻ Replaced existing entry for id={entry_id}")
                            break

                if not replaced_existing:
                    sft_dataset.append(sft_entry)
                    if entry_id is not None:
                        existing_ids.add(entry_id)
                    new_successful_count += 1

                # 实时保存增量结果（无论新增还是替换）
                save_json_file(incremental_output_path, sft_dataset)
                save_json_file(incremental_reflection_path, sft_dataset)
                print(
                    f"  ✓ Incremental save to {incremental_output_path} ({len(sft_dataset)} entries)"
                )
            else:
                new_failed_count += 1

        print(f"\n{'=' * 60}")
        print(f"Summary (This Run):")
        print(
            f"  Newly solved samples (100% pass rate): {new_successful_count}/{len(unsolved_hard_samples)}"
        )
        print(
            f"  Still failed samples: {new_failed_count}/{len(unsolved_hard_samples)}"
        )
        if len(unsolved_hard_samples) > 0:
            print(
                f"  Success rate (this run): {new_successful_count / len(unsolved_hard_samples):.1%}"
            )
        print(f"{'=' * 60}")
    else:
        print(f"\nNo unsolved hard samples to process. All samples are already solved!")
        new_successful_count = 0
        new_failed_count = 0

    # 移除 total_steps=1 的 hard 样本
    removed_total_steps_one_count = 0
    removed_total_steps_one_ids = []
    filtered_dataset = []
    for entry in sft_dataset:
        if entry.get("difficulty") == "hard":
            reflection_data = entry.get("reflection_data") or {}
            best_reflection = (
                reflection_data.get("best_reflection") if reflection_data else None
            )
            if best_reflection and best_reflection.get(
                "success"
            ):  # 只针对成功的 reflection
                total_steps = best_reflection.get("total_steps", 0)
                if total_steps <= 1:
                    removed_total_steps_one_count += 1
                    entry_id = entry.get("id")
                    if entry_id is not None:
                        removed_total_steps_one_ids.append(entry_id)
                    continue
        filtered_dataset.append(entry)

    if removed_total_steps_one_count > 0:
        print(
            f"\nRemoved {removed_total_steps_one_count} hard samples with total_steps <= 1 from final dataset"
        )

    sft_dataset = filtered_dataset

    # 计算总体统计
    total_hard_entries = [e for e in sft_dataset if e.get("difficulty") == "hard"]
    total_solved_hard = len(
        [
            e
            for e in total_hard_entries
            if e.get("reflection_data", {})
            .get("best_reflection", {})
            .get("success", False)
            or e.get("reflection_data", {}).get("best_pass_rate", 0.0) == 1.0
            or e.get("reflection_data", {}).get("reason")
            == "all_codes_passed_or_no_codes"
        ]
    )

    print(f"\n{'=' * 60}")
    print(f"Overall Summary:")
    print(f"  Total hard samples in dataset: {len(total_hard_entries)}")
    print(f"  Total solved hard samples: {total_solved_hard}")
    print(f"  Total hard samples (from input): {len(hard_samples)}")
    print(
        f"  Overall solve rate: {total_solved_hard / len(hard_samples):.1%}"
        if len(hard_samples) > 0
        else "  N/A"
    )
    print(f"{'=' * 60}")

    # 保存 SFT 数据集
    print(f"\nSaving SFT dataset to {output_path}")
    save_json_file(output_path, sft_dataset)

    # 保存完整 reflection 数据
    print(f"Saving reflection data to {reflection_data_path}")
    save_json_file(reflection_data_path, sft_dataset)

    # 同步更新增量文件为最终版本
    save_json_file(incremental_output_path, sft_dataset)
    save_json_file(incremental_reflection_path, sft_dataset)

    removed_total_steps_one = removed_total_steps_one_count

    # 同步更新 hard 输入文件，移除 total_steps <= 1 的条目
    if removed_total_steps_one_ids:
        ids_to_remove = set(removed_total_steps_one_ids)
        print(
            f"Updating hard input file by removing {len(ids_to_remove)} entries with total_steps <= 1"
        )
        try:
            with open(hard_input_path, "r") as f:
                try:
                    hard_dataset_full = json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)
                    hard_dataset_full = [json.loads(line) for line in f if line.strip()]

            original_len = len(hard_dataset_full)
            updated_hard_dataset = [
                sample
                for sample in hard_dataset_full
                if sample.get("id") not in ids_to_remove
            ]
            removed_from_input = original_len - len(updated_hard_dataset)

            if removed_from_input != len(ids_to_remove):
                print(
                    f"  Warning: Expected to remove {len(ids_to_remove)} entries but removed {removed_from_input}"
                )

            save_json_file(hard_input_path, updated_hard_dataset)

            print(f"  Updated hard input file: {removed_from_input} entries removed")
        except Exception as e:
            print(f"  Warning: Failed to update hard input file: {e}")

    print("\n" + "=" * 60)
    print("COMPLETED!")
    print("=" * 60)
    print(f"\nFinal Statistics:")
    print(f"  Total samples in dataset: {len(sft_dataset)}")
    print(
        f"  Hard samples: {len([e for e in sft_dataset if e.get('difficulty') == 'hard'])}"
    )
    print(f"  Newly solved in this run: {new_successful_count}")
    print(f"  Already solved (from previous runs): {len(existing_solved_ids)}")
    if removed_total_steps_one > 0:
        print(f"  Removed (total_steps <= 1): {removed_total_steps_one}")
    print(f"\nOutput files:")
    print(f"  SFT dataset: {output_path}")
    print(f"  Reflection data: {reflection_data_path}")


if __name__ == "__main__":
    fire.Fire(main)
