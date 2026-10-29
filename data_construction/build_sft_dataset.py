import json
import fire
from typing import List, Dict, Any
from tqdm import tqdm

from openrlhf.cli.eval_test_cases import evaluate


def extract_qwen_coder_completions(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从 inferences 中提取 qwen_coder_2.5 模型的 completions

    Args:
        sample: 单个样本数据

    Returns:
        包含 completion 和 completion_id 的字典列表
    """
    qwen_completions = []
    inferences = sample.get("inferences", [])

    for inference in inferences:
        if inference.get("model_name") == "qwen_coder_2.5":
            qwen_completions.append(
                {
                    "completion": inference.get("completion", ""),
                    "completion_id": inference.get("completion_id", -1),
                    "original_pass_rate": inference.get("pass_rate", 0.0),
                }
            )

    return qwen_completions


def prepare_eval_samples(
    sample: Dict[str, Any], completions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    准备用于评估的样本列表

    Args:
        sample: 原始样本数据
        completions: qwen_coder_2.5 的 completions 列表

    Returns:
        格式化后的评估样本列表
    """
    eval_samples = []
    task_id = sample.get("id", "unknown")
    test_cases = sample.get("test_cases", [])

    for comp_data in completions:
        eval_sample = {
            "task_id": task_id,
            "output": comp_data["completion"],
            "tests": test_cases,
            "_identifier": f"{task_id}_{comp_data['completion_id']}",
        }
        eval_samples.append(eval_sample)

    return eval_samples


def build_sft_entry(
    sample: Dict[str, Any], difficulty: str, eval_results: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    构建单个 SFT 训练样本

    Args:
        sample: 原始样本数据
        difficulty: 'easy' 或 'hard'
        eval_results: 评估结果列表（仅 hard 需要）

    Returns:
        SFT 格式的训练样本
    """
    # 提取 prompt (question 文本)
    question = sample.get("question", "")

    sft_entry = {
        "id": sample.get("id"),
        "source": sample.get("source"),
        "difficulty": difficulty,
        "prompt": question,
    }

    if difficulty == "easy":
        # easy 题目：output 为 <eos>
        sft_entry["output"] = "<eos>"
    else:
        # hard 题目：需要从评估结果中选择最佳 completion
        # 这里先占位，后续实现选择逻辑
        sft_entry["eval_results"] = eval_results
        sft_entry["output"] = None  # 待实现

    return sft_entry


def main(
    hard_input_path: str = "./data/acecode_87K/acecode_87k_hard_sft.json",
    easy_input_path: str = "./data/acecode_87K/acecode_87k_easy_sft.json",
    output_path: str = "./data/acecode_87K/acecode_sft_formatted.json",
    n_samples: int = 20,
    n_workers: int = 16,
):
    """
    构建标准 SFT 数据集

    Args:
        hard_input_path: hard 样本输入路径
        easy_input_path: easy 样本输入路径
        output_path: 输出 SFT 数据集路径
        n_samples: 处理的样本数量（用于测试，0 表示全部）
        n_workers: 评估时的并行 worker 数
    """
    # 加载 hard 数据
    print(f"Loading hard samples from {hard_input_path}")
    with open(hard_input_path, "r") as f:
        try:
            hard_samples = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            hard_samples = [json.loads(line) for line in f if line.strip()]

    # 加载 easy 数据
    print(f"Loading easy samples from {easy_input_path}")
    with open(easy_input_path, "r") as f:
        try:
            easy_samples = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            easy_samples = [json.loads(line) for line in f if line.strip()]

    print(
        f"Loaded {len(hard_samples)} hard samples and {len(easy_samples)} easy samples"
    )

    # 限制样本数量用于测试
    if n_samples > 0:
        hard_samples = hard_samples[:20]
        easy_samples = easy_samples[:20]
        print(
            f"Processing first {len(hard_samples)} hard + {len(easy_samples)} easy samples for testing"
        )

    sft_dataset = []

    print(f"\nProcessing {len(easy_samples)} easy samples...")
    for sample in tqdm(easy_samples, desc="Processing easy samples"):
        sft_entry = build_sft_entry(sample, "easy")
        sft_dataset.append(sft_entry)

    print(f"\nProcessing {len(hard_samples)} hard samples...")
    print("Step 1: Extracting qwen_coder_2.5 completions...")

    # 收集所有需要评估的样本
    all_eval_samples = []
    hard_sample_mapping = []  # 记录每个评估样本对应的原始 hard sample 索引

    for idx, sample in enumerate(tqdm(hard_samples, desc="Extracting completions")):
        qwen_completions = extract_qwen_coder_completions(sample)
        if len(qwen_completions) > 0:
            eval_samples = prepare_eval_samples(sample, qwen_completions)
            all_eval_samples.extend(eval_samples)
            # 记录映射关系
            for _ in eval_samples:
                hard_sample_mapping.append(idx)
        else:
            print(
                f"Warning: No qwen_coder_2.5 completions found for sample {sample.get('id')}"
            )

    print(f"\nStep 2: Evaluating {len(all_eval_samples)} completions...")
    if len(all_eval_samples) > 0:
        # 调用评估函数
        eval_results, pass_rates = evaluate(
            samples=all_eval_samples,
            dataset=None,
            base_only=True,
            n_workers=n_workers,
            test_details=True,
            extract_solution=True,
            output_file=None,
        )
        print(f"\nStep 3: Building SFT entries for hard samples...")
        # 将评估结果按原始 sample 分组
        sample_eval_results = {}
        for eval_idx, eval_result in enumerate(eval_results):
            original_sample_idx = hard_sample_mapping[eval_idx]
            if original_sample_idx not in sample_eval_results:
                sample_eval_results[original_sample_idx] = []
            sample_eval_results[original_sample_idx].append(eval_result)

        # 为每个 hard sample 构建 SFT entry
        for idx, sample in enumerate(hard_samples):
            results = sample_eval_results.get(idx, [])
            sft_entry = build_sft_entry(sample, "hard", results)
            sft_dataset.append(sft_entry)

    # 保存结果
    print(f"\nSaving {len(sft_dataset)} SFT samples to {output_path}")
    with open(output_path, "w") as f:
        json.dump(sft_dataset, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved to {output_path}")

    # 打印统计信息
    easy_count = sum(1 for s in sft_dataset if s["difficulty"] == "easy")
    hard_count = sum(1 for s in sft_dataset if s["difficulty"] == "hard")
    print(f"\nStatistics:")
    print(f"  Easy samples: {easy_count}")
    print(f"  Hard samples: {hard_count}")
    print(f"  Total: {len(sft_dataset)}")


if __name__ == "__main__":
    fire.Fire(main)
