import fire
import datasets
import numpy as np

r1_system_prompt = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
"""


def main(
    dataset_path="CodeDPO/codedpo_20241208",
    output_path="CodeDPO/codedpo_20241208_openrlhf_format",
    only_keep_hard_examples=True,
    add_r1_system_prompt=False,
):
    dataset = datasets.load_dataset(dataset_path, split="train")
    print(f"Loaded {len(dataset)} examples")

    def map_openrlhf_format(item):
        item["context_messages"] = [{"content": item["question"], "role": "user"}]
        return item

    dataset = dataset.map(
        map_openrlhf_format, desc="Mapping to OpenRLHF format", num_proc=4
    )

    if only_keep_hard_examples:
        output_path += "_hard"
        consider_models = ["qwen_coder_2.5", "llama3_instruct"]

        def get_accs(item):
            inference = eval(item["inference"])
            inference = [x for x in inference if x[2] in consider_models]
            accs = [x[1] for x in inference]
            item["accs"] = accs
            return item

        dataset = dataset.map(get_accs, desc="Getting accs", num_proc=4)

        # then get the lower 50% of the low average examples
        avg_accs = [np.mean(x) for x in dataset["accs"]]
        split_acc = np.percentile(avg_accs, 50)
        print(f"Splitting by acc at {split_acc}")
        dataset = dataset.filter(
            lambda x: np.mean(x["accs"]) <= split_acc,
            desc="Filtering low average examples",
        )
        print(f"Kept {len(dataset)} examples")
        for i in range(3):
            print(dataset[i]["accs"])

        # first get the top 50% of the high std examples
        std_accs = [np.std(x) for x in dataset["accs"]]
        split_std = np.percentile(std_accs, 50)
        print(f"Splitting by std at {split_std}")
        dataset = dataset.filter(
            lambda x: np.std(x["accs"]) >= split_std, desc="Filtering high std examples"
        )
        print(f"Kept {len(dataset)} examples")
        for i in range(3):
            print(dataset[i]["accs"])

        dataset = dataset.remove_columns("accs")

    if add_r1_system_prompt:

        def add_r1_prompt(item):
            item["context_messages"].insert(
                0, {"content": r1_system_prompt, "role": "system"}
            )
            return item

        dataset = dataset.map(add_r1_prompt, desc="Adding R1 system prompt", num_proc=4)
        output_path += "_r1"

    dataset.push_to_hub(output_path)
    print(f"Pushed to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)


"""
python scripts/format_code_rl_data.py --dataset_path "CodeDPO/codedpo_20241208" --output_path "CodeDPO/codedpo_20241208_openrlhf_format" --only_keep_hard_examples True
python scripts/format_code_rl_data.py --dataset_path "CodeDPO/rlhf_dataset_20250126" --output_path "CodeDPO/rlhf_dataset_20250126_openrlhf_format" --only_keep_hard_examples True
python scripts/format_code_rl_data.py --dataset_path "CodeDPO/rlhf_dataset_20250126" --output_path "CodeDPO/rlhf_dataset_20250126_openrlhf_format" --only_keep_hard_examples True --add_r1_system_prompt True
"""
