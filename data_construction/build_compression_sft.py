"""
构建 Compression-based SFT 数据集

工作流:
1. 从 Reflection 数据集读取已解决的 cases
2. 对每个 reflection 生成编程 tips
3. 将 tips 追加到原始 instruction 后
4. 用增强的 instruction 生成新代码
5. 执行测试验证新代码质量
"""

import json
import fire
import os
import re
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from openai import OpenAI
from evalplus.sanitize import code_extract

# Add parent directory to path
from openrlhf.cli.eval_test_cases import evaluate


def save_json_file(path: str, data: Any) -> None:
    """Helper to write JSON with UTF-8 encoding."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_skip_cases(skip_file_path: str) -> set:
    """加载需要跳过的case ID集合"""
    if os.path.exists(skip_file_path):
        try:
            with open(skip_file_path, "r") as f:
                skip_data = json.load(f)
                return set(skip_data.keys())
        except Exception as e:
            print(f"Warning: Failed to load skip cases from {skip_file_path}: {e}")
            return set()
    return set()


def add_skip_case(skip_file_path: str, case_id: str, reason: str):
    """添加需要跳过的case"""
    skip_data = {}
    if os.path.exists(skip_file_path):
        try:
            with open(skip_file_path, "r") as f:
                skip_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load existing skip data: {e}")
            skip_data = {}

    skip_data[case_id] = reason
    try:
        with open(skip_file_path, "w") as f:
            json.dump(skip_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to save skip case {case_id}: {e}")


def execute_and_get_feedback(
    code: str, task_id: str, test_cases: List[str], n_workers: int = 8
) -> Dict[str, Any]:
    """
    执行代码并获取详细的 feedback
    """
    eval_sample = {
        "task_id": task_id,
        "output": code,
        "tests": test_cases,
        "_identifier": f"{task_id}_feedback",
    }

    # 首先尝试直接执行代码来捕获语法错误等
    error_info = None
    try:
        # 使用 evalplus 的代码提取和清理功能
        cleaned_code = code_extract(
            code.encode("utf-8", "ignore").decode("utf-8").replace("\x00", "")
        )
        if not cleaned_code or not cleaned_code.strip():
            cleaned_code = code

        # 尝试编译代码来检查语法错误
        compile(cleaned_code, "<string>", "exec")
    except SyntaxError as e:
        error_info = f"SyntaxError: {e.msg} at line {e.lineno}"
    except Exception as e:
        error_info = f"Code compilation error: {str(e)}"

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

    # 如果没有语法错误但所有测试都失败了，可能有运行时错误
    if error_info is None and len(passed_tests) > 0 and not any(passed_tests):
        status = result["base"][0]  # 状态字符串
        if status == "fail":
            error_info = (
                "Runtime error: All test cases failed, likely due to execution errors"
            )
        elif status == "timeout":
            error_info = "Timeout: Code execution timed out"
        elif status:
            error_info = f"Execution error: {status}"

    return {
        "pass_rate": result["pass_rate"],
        "passed_tests": passed_tests,
        "failed_indices": failed_indices,
        "total_tests": len(passed_tests),
        "solution": result["solution"],
        "error_info": error_info,
    }


def generate_tips_prompt(reflection_data: Dict[str, Any]) -> Tuple[str, str]:
    """
    生成从 reflection 中提取编程 tips 的提示
    """
    best_reflection = reflection_data.get("best_reflection", {})
    reflection_chain = best_reflection.get("reflection_chain", [])

    # 构建对话历史，mask掉代码和测试文件
    dialogue_summary = ""
    for i, entry in enumerate(reflection_chain):
        step = entry.get("step", 0)
        dialogue = entry.get("dialogue", [])

        dialogue_summary += f"\n=== Step {step + 1} ===\n"

        for msg in dialogue:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Mask 掉代码块和测试相关内容
            # 移除 ```python ... ``` 代码块
            content = re.sub(
                r"```python\s*\n.*?```", "[CODE_BLOCK_MASKED]", content, flags=re.DOTALL
            )
            # 移除 ``` ... ``` 代码块
            content = re.sub(
                r"```\s*\n.*?```", "[CODE_BLOCK_MASKED]", content, flags=re.DOTALL
            )
            # 替换整个测试用例部分
            content = re.sub(
                r"Failed Test Cases:.*",
                "Failed Test Cases:\n[TEST_CASES_MASKED]",
                content,
                flags=re.DOTALL,
            )
            # 移除 def test_ 开头的函数
            content = re.sub(
                r"def test_\w+.*?(?=def|\n\n|\Z)",
                "[TEST_FUNCTION_MASKED]",
                content,
                flags=re.DOTALL,
            )

            dialogue_summary += f"{role.upper()}: {content.strip()}\n"

    prompt = f"""You are analyzing a programming debugging conversation where a human and an AI assistant are working together to fix code issues. The conversation has been masked to hide specific code implementations and test cases.

**Debugging Conversation:**
{dialogue_summary}

**Task:**
Based on this debugging conversation, extract and summarize the key programming tips and insights that would help someone solve this coding problems in the future. Focus on:

1. **Problem Analysis**: What types of issues were identified?
2. **Missing Requirements**: What requirements were ignored or not met?
3. **Best Practices**: What coding principles or patterns emerged?
4. **Common Pitfalls**: What mistakes were made and how to avoid them?

Provide a concise summary of key tips that capture the essence of this debugging process. Write it as a itemized list that could be appended to a coding problem instruction.
Do not provide any solutions, code snippets, test/verify/debugging instructions, or any general programming principles that are not specific in the tips.

Try to be brief and concise. Too long tips can also burden the LLM's memory.

**Output Format:**
Just provide the tips paragraph, no additional formatting or explanations.
"""

    return prompt, dialogue_summary


def call_openai_api(
    prompt: str,
    api_key: str,
    api_base: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    system_message: str = "You are a helpful coding assistant.",
    max_retries: int = 2,
    retry_delay: float = 30.0,
) -> str:
    """
    调用 OpenAI API 生成改进的代码（带重试机制）
    """
    # 创建客户端
    client = OpenAI(api_key=api_key, base_url=api_base, timeout=240)

    last_error = None

    for attempt in range(max_retries):
        try:
            print("  >> Sending request to OpenAI API...")
            # print(f"     model: {model}, temperature: {temperature}")
            # print("----- LLM Request Start -----")
            # print(prompt)
            # print("----- LLM Request End -----")

            # 构建消息列表
            messages = []
            if system_message and system_message.strip():
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=model,
                messages=messages,
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
            import time

            time.sleep(retry_delay * (attempt + 1))

    # 所有重试都失败
    raise Exception(
        f"API call failed after {max_retries} attempts. Last error: {last_error}"
    )


def call_vllm_api(
    prompt: str,
    vllm_url: str = "http://localhost:8000/v1",
    model: str = "weak_model",
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout: float = 30.0,
) -> str:
    """
    调用 vLLM API 生成代码
    """
    import requests

    try:
        response = requests.post(
            f"{vllm_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(
                f"vLLM API returned status {response.status_code}: {response.text}"
            )

    except Exception as e:
        raise Exception(f"vLLM API call failed: {e}")


def generate_code_with_tips(
    original_instruction: str,
    tips: str,
    vllm_url: str = "http://localhost:8000/v1",
    model: str = "weak_model",
) -> str:
    """
    使用增强的instruction（原始instruction + tips）生成代码
    """
    enhanced_instruction = f"{original_instruction}\n{tips}"

    try:
        response = call_vllm_api(
            prompt=enhanced_instruction,
            vllm_url=vllm_url,
            model=model,
            temperature=0.1,  # 更低的温度以获得更稳定的输出
            max_tokens=2048,
        )

        # 提取代码
        try:
            extracted_code = code_extract(
                response.encode("utf-8", "ignore").decode("utf-8").replace("\x00", "")
            )
            if extracted_code and extracted_code.strip():
                return extracted_code
            else:
                return response
        except Exception as e:
            print(f"  Code extraction failed: {e}, using raw response")
            return response

    except Exception as e:
        print(f"  Failed to generate code: {e}")
        return ""


def process_reflection_sample(
    reflection_entry: Dict[str, Any],
    api_key: str,
    api_base: str,
    vllm_url: str = "http://localhost:8000/v1",
    vllm_model: str = "weak_model",
    n_workers: int = 8,
    openai_model: str = "gpt-4o-mini",
    skip_cases_path: str = "./data/acecode_87K/compression_skip_cases.json",
) -> Dict[str, Any]:
    """
    处理单个 reflection 样本：生成 tips -> 生成代码 -> 测试验证
    """
    task_id = reflection_entry.get("id", "unknown")
    original_instruction = reflection_entry.get("prompt", "")
    reflection_data = reflection_entry.get("reflection_data", {})
    test_cases = reflection_entry.get("test_cases", [])

    print(f"Processing reflection sample {task_id}")

    # 获取原数据集中的模型统计信息，用于后续输出
    original_stats = None
    inferences = reflection_entry.get("original_inferences", [])
    if inferences:
        pass_rates = [inf.get("pass_rate", 0.0) for inf in inferences]
        if pass_rates:
            original_stats = {
                "count": len(pass_rates),
                "mean": sum(pass_rates) / len(pass_rates),
                "min": min(pass_rates),
                "max": max(pass_rates),
                "median": sorted(pass_rates)[len(pass_rates) // 2],
            }
            print(
                f"  Found {len(inferences)} original inferences with pass_rates: {pass_rates[:3]}..."
            )
        else:
            print(f"  Found {len(inferences)} original inferences but no pass_rates")
    else:
        print(f"  No original_inferences found in reflection_entry")

    # 检查是否有有效的reflection数据
    best_reflection = reflection_data.get("best_reflection")
    if not best_reflection or not best_reflection.get("success", False):
        print(f"  No valid reflection data, skipping permanently")
        add_skip_case(skip_cases_path, task_id, "invalid_reflection_data")
        return None

    total_steps = best_reflection.get("total_steps", 0)
    if total_steps <= 1:
        print(f"  Reflection has total_steps={total_steps}, skipping permanently")
        add_skip_case(skip_cases_path, task_id, "insufficient_reflection_steps")
        return None

    # Step 1: 生成编程tips
    print(f"  Step 1: Generating programming tips...")
    tips_prompt, dialogue_summary = generate_tips_prompt(reflection_data)
    try:
        tips = call_openai_api(
            prompt=tips_prompt,
            api_key=api_key,
            api_base=api_base,
            model=openai_model,
            temperature=0.3,  # 稍微稳定的输出
            system_message="You are a helpful coding assistant that extracts programming tips from debugging conversations.",
        )
        print(f"    Generated tips ({len(tips)} chars): {tips}")
        print("=" * 60)
    except Exception as e:
        print(f"  Failed to generate tips: {e}")
        return None

    # Step 2: 使用增强的instruction生成代码
    print(f"  Step 2: Generating code with tips...")
    try:
        generated_code = generate_code_with_tips(
            original_instruction=original_instruction,
            tips=tips,
            vllm_url=vllm_url,
            model=vllm_model,
        )
        if not generated_code or not generated_code.strip():
            print(f"  Generated empty code, skipping permanently")
            add_skip_case(skip_cases_path, task_id, "empty_code_generation")
            return None
        print(f"    Generated code ({len(generated_code)} chars)")
        print(f"    Full generated code:")
        print("-" * 40)
        print(generated_code)
        print("-" * 40)
    except Exception as e:
        print(f"  Failed to generate code: {e}")
        return None

    # Step 3: 测试验证生成的代码
    print(f"  Step 3: Testing generated code...")

    # 初始化变量用于retry逻辑
    current_tips = tips
    current_generated_code = generated_code
    retry_count = 0
    max_retries = 2  # 设置最大重试次数为1

    while retry_count <= max_retries:
        try:
            test_feedback = execute_and_get_feedback(
                code=current_generated_code,
                task_id=task_id,
                test_cases=test_cases,
                n_workers=n_workers,
            )

            pass_rate = test_feedback["pass_rate"]
            print(
                f"    Test results: {pass_rate:.2%} pass rate ({len(test_feedback['failed_indices'])} failed)"
            )

            # 输出原始模型统计信息
            if original_stats:
                print("    Original model stats (qwen_coder_2.5):")
                print(
                    f"      count: {original_stats['count']}, mean: {original_stats['mean']:.2%}, min: {original_stats['min']:.2%}, max: {original_stats['max']:.2%}, median: {original_stats['median']:.2%}"
                )

            # 如果完全通过，保存结果
            if pass_rate == 1.0:
                print(
                    f"  ✓ Success! Generated code passes all tests{' (after retry)' if retry_count > 0 else ''}"
                )

                return {
                    "id": task_id,
                    "source": reflection_entry.get("source"),
                    "prompt": f"{original_instruction}\n\nProgramming Tips: {current_tips}",
                    "output": current_generated_code,
                    "original_instruction": original_instruction,
                    "tips": current_tips,
                    "generated_code": current_generated_code,
                    "test_pass_rate": pass_rate,
                    "reflection_total_steps": total_steps,
                    "baseline_stats": original_stats,
                    "retry_count": retry_count,
                }

            # 如果测试失败且还有重试次数，尝试改进tips
            if retry_count < max_retries:
                # 检查是否是网络相关错误，如果是则不计入retry次数
                error_info = test_feedback.get("error_info", "")
                is_network_error = error_info and any(
                    keyword in error_info.lower()
                    for keyword in [
                        "timeout",
                        "connection",
                        "network",
                        "api",
                        "http",
                        "server",
                    ]
                )

                if is_network_error:
                    print(
                        f"  ⚠ Network error detected ({error_info}), retrying without counting towards retry limit"
                    )
                    # 不增加retry_count，继续重试
                else:
                    print(
                        f"  ⚠ Test failed (pass_rate: {pass_rate:.2%}), attempting to improve tips (retry {retry_count + 1}/{max_retries})"
                    )
                    retry_count += 1

                # 根据失败类型生成不同的feedback
                num_failed = len(test_feedback["failed_indices"])

                print(f"    Debug: num_failed={num_failed}, error_info='{error_info}'")

                improve_tips_prompt = f"""You are given a coding problem, some programming tips, and feedback about why the generated code failed.

**Problem:**
{original_instruction}

**Current Programming Tips:**
{current_tips}

**Debugging Conversation:**
{dialogue_summary}
"""

                # 如果是第二次retry (retry_count == 1)，提供正确代码作为参考
                if retry_count == 1:
                    # 获取正确代码 - 从reflection_chain的最后一项
                    reflection_chain = best_reflection.get("reflection_chain", [])
                    if reflection_chain:
                        correct_code = reflection_chain[-1].get("code", "")
                        if correct_code:
                            improve_tips_prompt += f"""
**Correct Reference Code:**
```python
{correct_code}
```
"""

                if num_failed == 0:
                    # 代码无法运行的情况 - 提供错误信息
                    if error_info and error_info != "None" and error_info.strip():
                        error_msg = error_info
                    else:
                        error_msg = (
                            "Syntax or compilation error: Code could not be executed"
                        )

                    improve_tips_prompt += f"""
**Code Execution Error:**
{error_msg}

**Task:**
The code generated using the current tips could not run due to the above error. Please analyze what programming tips would help generate syntactically correct and executable code for this problem.
Try to be brief and concise. Too long tips can also burden the LLM's memory.
Provide concise, actionable programming tips that would help generate correct, runnable code.
"""
                else:
                    # 代码能运行但测试失败的情况 - 提供失败的测试用例
                    failed_tests = [
                        test_cases[i] for i in test_feedback["failed_indices"]
                    ]
                    improve_tips_prompt += f"""
**Failed Test Cases:**
"""
                    for i, test in enumerate(failed_tests, 1):
                        improve_tips_prompt += f"{i}. {test}\n"

                    improve_tips_prompt += """
**Task:**
The code generated using the current tips failed these specific test cases. Please analyze why the tips were insufficient and provide IMPROVED programming tips that would help generate correct code for this problem.
Try to be brief and concise. Too long tips can also burden the LLM's memory.
Provide concise, actionable programming tips that could be appended to the original instruction to guide code generation.
"""

                improve_tips_prompt += """
**Output Format:**
Just provide the improved tips paragraph, no additional formatting or explanations.
"""

                try:
                    print("    Improving tips...")
                    improved_tips = call_openai_api(
                        prompt=improve_tips_prompt,
                        api_key=api_key,
                        api_base=api_base,
                        model=openai_model,
                        temperature=0.3,
                        system_message="You are a helpful coding assistant that improves programming tips based on failed test cases.",
                    )
                    current_tips = improved_tips
                    print(
                        f"    Improved tips ({len(improved_tips)} chars): {improved_tips}"
                    )

                    # 使用改进的tips重新生成代码
                    print("    Regenerating code with improved tips...")
                    current_generated_code = generate_code_with_tips(
                        original_instruction=original_instruction,
                        tips=current_tips,
                        vllm_url=vllm_url,
                        model=vllm_model,
                    )
                    print(
                        f"      Regenerated code ({len(current_generated_code)} chars)"
                    )
                    print(f"      Full regenerated code:")
                    print("-" * 40)
                    print(current_generated_code)
                    print("-" * 40)

                except Exception as e:
                    print(f"    Failed to improve tips: {e}")
                    break

                retry_count += 1
            else:
                # 检查最后一次失败是否是网络错误
                final_error_info = test_feedback.get("error_info", "")
                is_final_network_error = final_error_info and any(
                    keyword in final_error_info.lower()
                    for keyword in [
                        "timeout",
                        "connection",
                        "network",
                        "api",
                        "http",
                        "server",
                    ]
                )

                if is_final_network_error:
                    print(
                        f"  ✗ Final attempt failed due to network error ({final_error_info}), will retry in next run"
                    )
                    return None
                else:
                    print(
                        f"  ✗ Generated code failed tests (pass_rate: {pass_rate:.2%}) after {retry_count} retries, skipping permanently"
                    )
                    add_skip_case(skip_cases_path, task_id, "persistent_test_failure")
                    return None

        except Exception as e:
            print(f"  Failed to test code: {e}")
            # 对于测试失败，如果不是明显的网络错误，写入skip文件
            error_msg = str(e).lower()
            if (
                "timeout" not in error_msg
                and "connection" not in error_msg
                and "network" not in error_msg
            ):
                add_skip_case(skip_cases_path, task_id, "testing_execution_error")
            return None


def main(
    reflection_data_path: str = "./data/acecode_87K/acecode_reflection_data.json",
    output_path: str = "./data/acecode_87K/acecode_compression_sft.json",
    compression_data_path: str = "./data/acecode_87K/acecode_compression_data.json",
    skip_cases_path: str = "./data/acecode_87K/compression_skip_cases.json",
    api_key: str = None,
    api_base: str = "https://api.openai.com/v1",
    vllm_url: str = "http://localhost:30000/v1",
    vllm_model: str = "./models/model",
    n_samples: int = 0,
    n_workers: int = 8,
    openai_model: str = "gemini-3-pro-all",
):
    """
    构建 Compression-based SFT 数据集

    Args:
        reflection_data_path: reflection 数据输入路径
        output_path: 输出 SFT 数据集路径
        compression_data_path: 保存完整 compression 数据的路径
        skip_cases_path: 记录需要跳过case的文件路径
        api_key: OpenAI API key
        api_base: OpenAI API base URL
        vllm_url: vLLM 服务 URL
        vllm_model: vLLM 模型名称
        n_samples: 处理的样本数量（0 表示全部）
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
    print("Compression-based SFT Dataset Builder")
    print("=" * 60)
    print(f"Reflection input: {reflection_data_path}")
    print(f"Output: {output_path}")
    print(f"Compression data: {compression_data_path}")
    print(f"Skip cases file: {skip_cases_path}")
    print(f"OpenAI model: {openai_model}")
    print(f"vLLM URL: {vllm_url}")
    print(f"vLLM model: {vllm_model}")
    print("=" * 60)

    # 加载数据
    print("\nLoading data...")
    with open(reflection_data_path, "r") as f:
        try:
            reflection_samples = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            reflection_samples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(reflection_samples)} reflection samples")

    # 加载需要跳过的cases
    print("Loading skip cases...")
    skip_case_ids = load_skip_cases(skip_cases_path)
    print(f"Found {len(skip_case_ids)} cases to skip")

    # 同时加载原始hard数据以获取test_cases
    hard_input_path = "./data/acecode_87K/acecode_87k_hard_sft_filtered.json"
    with open(hard_input_path, "r") as f:
        try:
            hard_samples = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            hard_samples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(hard_samples)} hard samples for test cases")

    # 创建ID到test_cases和inferences的映射
    id_to_test_cases = {
        sample.get("id"): sample.get("test_cases", []) for sample in hard_samples
    }
    id_to_inferences = {
        sample.get("id"): sample.get("inferences", []) for sample in hard_samples
    }

    # 将test_cases和原始inferences添加到reflection样本中
    for sample in reflection_samples:
        sample_id = sample.get("id")
        if sample_id in id_to_test_cases:
            sample["test_cases"] = id_to_test_cases[sample_id]
        if sample_id in id_to_inferences:
            sample["original_inferences"] = id_to_inferences[sample_id]

    # 检查是否有已存在的结果文件（重试机制）
    existing_compression_ids = set()
    existing_compression_dataset = []

    # 检查主文件和partial文件
    files_to_check = [compression_data_path, compression_data_path + ".partial.json"]
    found_files = []

    for file_path in files_to_check:
        if os.path.exists(file_path):
            found_files.append(file_path)

    if found_files:
        print(f"\nFound existing compression data files: {found_files}")
        print("Loading existing results to identify already processed cases...")

        for file_path in found_files:
            try:
                with open(file_path, "r") as f:
                    file_data = json.load(f)

                print(f"  Loading {len(file_data)} entries from {file_path}")

                # 合并数据，避免重复
                for entry in file_data:
                    entry_id = entry.get("id")
                    if (
                        entry_id is not None
                        and entry_id not in existing_compression_ids
                    ):
                        existing_compression_dataset.append(entry)
                        existing_compression_ids.add(entry_id)

            except Exception as e:
                print(f"  Warning: Failed to load data from {file_path}: {e}")

        if existing_compression_dataset:
            print(
                f"  Total: Found {len(existing_compression_ids)} already processed cases"
            )
            print(
                f"  Total: Existing dataset contains {len(existing_compression_dataset)} entries"
            )
        else:
            print("  No valid data found in existing files")
            existing_compression_dataset = []
            existing_compression_ids = set()
    else:
        print(f"\nNo existing compression data found")
        print("Will process all samples from scratch")

    # 过滤出未处理的reflection样本
    unprocessed_reflection_samples = [
        sample
        for sample in reflection_samples
        if sample.get("id") not in existing_compression_ids
        and sample.get("id") not in skip_case_ids
    ]

    print(f"\nFiltering unprocessed cases:")
    print(f"  Total reflection samples: {len(reflection_samples)}")
    print(f"  Already processed: {len(existing_compression_ids)}")
    print(f"  Permanently skipped: {len(skip_case_ids)}")
    print(f"  Remaining unprocessed: {len(unprocessed_reflection_samples)}")

    # 限制样本数量用于测试
    if n_samples > 0:
        unprocessed_reflection_samples = unprocessed_reflection_samples[:n_samples]
        print(
            f"  Processing {len(unprocessed_reflection_samples)} samples for testing (n_samples={n_samples})"
        )

    # 初始化compression_dataset：先加载已有的已处理样本
    compression_dataset = (
        existing_compression_dataset.copy() if existing_compression_dataset else []
    )

    incremental_output_path = output_path + ".partial.json"
    incremental_compression_path = compression_data_path + ".partial.json"

    # 处理 reflection 样本
    if len(unprocessed_reflection_samples) > 0:
        print(
            f"\nProcessing {len(unprocessed_reflection_samples)} reflection samples with compression..."
        )
        new_successful_count = 0
        new_failed_count = 0

        for idx, sample in enumerate(
            tqdm(unprocessed_reflection_samples, desc="Processing reflection samples")
        ):
            print(
                f"\n[{idx + 1}/{len(unprocessed_reflection_samples)}] Processing sample {sample.get('id')}"
            )

            compression_entry = process_reflection_sample(
                reflection_entry=sample,
                api_key=api_key,
                api_base=api_base,
                vllm_url=vllm_url,
                vllm_model=vllm_model,
                n_workers=n_workers,
                openai_model=openai_model,
                skip_cases_path=skip_cases_path,
            )

            if compression_entry is not None:
                compression_dataset.append(compression_entry)
                new_successful_count += 1

                # 实时保存增量结果
                save_json_file(incremental_output_path, compression_dataset)
                save_json_file(incremental_compression_path, compression_dataset)
                print(
                    f"  ✓ Incremental save to {incremental_output_path} ({len(compression_dataset)} entries)"
                )
            else:
                new_failed_count += 1

        # 计算重试统计
        retry_counts = [
            entry.get("retry_count", 0)
            for entry in compression_dataset
            if entry.get("retry_count", 0) > 0
        ]
        total_retries = sum(retry_counts)
        samples_with_retries = len(retry_counts)

        # 计算retry前后成功的样本数量
        success_first_try = sum(
            1 for entry in compression_dataset if entry.get("retry_count", 0) == 0
        )
        success_after_retry = sum(
            1 for entry in compression_dataset if entry.get("retry_count", 0) > 0
        )

        print(f"\n{'=' * 60}")
        print(f"Summary (This Run):")
        print(
            f"  Successfully compressed samples: {new_successful_count}/{len(unprocessed_reflection_samples)}"
        )
        print(
            f"  Failed samples: {new_failed_count}/{len(unprocessed_reflection_samples)}"
        )
        if len(unprocessed_reflection_samples) > 0:
            print(
                f"  Success rate (this run): {new_successful_count / len(unprocessed_reflection_samples):.1%}"
            )

        # 显示retry统计
        if new_successful_count > 0:
            print(f"  Success breakdown:")
            print(f"    ✓ First try: {success_first_try} samples")
            print(f"    🔄 After retry: {success_after_retry} samples")
            if success_after_retry > 0:
                print(
                    f"    📈 Retry contribution: {success_after_retry / new_successful_count:.1%} of successes"
                )

        if samples_with_retries > 0:
            print(f"  Retry statistics:")
            print(f"    Samples that needed retries: {samples_with_retries}")
            print(f"    Total retry attempts: {total_retries}")
            print(
                f"    Average retries per failed sample: {total_retries / samples_with_retries:.1f}"
            )
        print(f"{'=' * 60}")
    else:
        print(
            f"\nNo unprocessed reflection samples to process. All samples are already compressed!"
        )
        new_successful_count = 0
        new_failed_count = 0

    # 计算总体统计
    total_compression_entries = [
        e for e in compression_dataset if e.get("difficulty") == "compression"
    ]
    total_processed = len(total_compression_entries)

    print(f"\n{'=' * 60}")
    print(f"Overall Summary:")
    print(f"  Total compression samples: {total_processed}")
    print(f"  Total reflection samples (from input): {len(reflection_samples)}")
    print(
        f"  Overall compression rate: {total_processed / len(reflection_samples):.1%}"
        if len(reflection_samples) > 0
        else "  N/A"
    )
    print(f"{'=' * 60}")

    # 保存 SFT 数据集
    print(f"\nSaving SFT dataset to {output_path}")
    save_json_file(output_path, compression_dataset)

    # 保存完整 compression 数据
    print(f"Saving compression data to {compression_data_path}")
    save_json_file(compression_data_path, compression_dataset)

    # 同步更新增量文件为最终版本
    save_json_file(incremental_output_path, compression_dataset)
    save_json_file(incremental_compression_path, compression_dataset)

    print("\n" + "=" * 60)
    print("COMPLETED!")
    print("=" * 60)
    print(f"\nFinal Statistics:")
    print(f"  Total samples in dataset: {len(compression_dataset)}")
    print(f"  Compression samples: {total_processed}")
    print(f"  Newly processed in this run: {new_successful_count}")
    print(f"  Already processed (from previous runs): {len(existing_compression_ids)}")
    print(f"  Permanently skipped: {len(skip_case_ids)}")
    print(f"\nOutput files:")
    print(f"  SFT dataset: {output_path}")
    print(f"  Compression data: {compression_data_path}")
    print(f"  Skip cases file: {skip_cases_path}")


if __name__ == "__main__":
    fire.Fire(main)
