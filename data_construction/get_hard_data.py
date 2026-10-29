import fire
import datasets
import json
import numpy as np
import os

r1_system_prompt = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
"""
def main(
    dataset_path,
    output_path="./data/acecode_87K/acecode_87k_v1.json",
    only_keep_hard_examples=True,
):
    if dataset_path.endswith('.json'):
        dataset = datasets.load_dataset('json', data_files=dataset_path, split='train')
        print(f"Loaded {len(dataset)} examples from local file: {dataset_path}")
    else:
        dataset = datasets.load_dataset(dataset_path, split="train")
        print(f"Loaded {len(dataset)} examples from HuggingFace: {dataset_path}")
    
    if only_keep_hard_examples:
        # output_path = output_path.replace(".json", "_hard.json")
        consider_models = ["qwen_coder_2.5"]
        def get_accs(item):
            inference = item['inferences']
            inference = [x for x in inference if x['model_name'] in consider_models]
            accs = [x['pass_rate'] for x in inference]
            item['accs'] = accs
            return item
        dataset = dataset.map(get_accs, desc="Getting accs", num_proc=4)

        # filter the all zero pass rate examples
        dataset = dataset.filter(lambda x: np.mean(x['accs']) > 0, desc="Filtering all zero pass rate examples")
        print(f"Kept {len(dataset)} examples")
        for i in range(3):
            print(dataset[i]['accs'])


        # # print some all zero std examples.
        # all_zero_std_examples = dataset.filter(lambda x: np.std(x['accs']) == 0 and np.mean(x['accs']) != 1)
        # print(f"Found {len(all_zero_std_examples)} all zero std examples")
        # for i in range(3):
        #     print(all_zero_std_examples[i]['accs'])
        # # and sample details
        # for i in range(3):
        #     print(all_zero_std_examples[i])

        # filter the all zero std examples, expect acc=1

        
        # dataset = dataset.filter(lambda x: np.std(x['accs']) > 0 and np.mean(x['accs']) != 1, desc="Filtering all zero std examples")
        # print(f"Kept {len(dataset)} examples") 

        
        # then get the lower 50% of the low average examples
        avg_accs = [np.mean(x) for x in dataset['accs']]
        split_acc = np.percentile(avg_accs, 50)
        print(f"Splitting by acc at {split_acc}")
        dataset_low_avg = dataset.filter(lambda x: np.mean(x['accs']) <= split_acc, desc="Filtering low average examples")
        print(f"Kept {len(dataset_low_avg)} examples")
        for i in range(3):
            print(dataset_low_avg[i]['accs'])
            
        # first get the top 50% of the high std examples
        std_accs = [np.std(x) for x in dataset_low_avg['accs']]
        split_std = np.percentile(std_accs, 50)
        print(f"Splitting by std at {split_std}")
        dataset_high_std = dataset_low_avg.filter(lambda x: np.std(x['accs']) >= split_std, desc="Filtering high std examples")
        print(f"Kept {len(dataset_high_std)} examples")
        for i in range(3):
            print(dataset_high_std[i]['accs'])

        # 记录已经在reflection中成功处理过的样本ID，这些样本不应该进入训练集
        # 优先使用partial文件，如果不存在则使用主文件
        reflection_data_ids = set()

        # 检查文件优先级：partial -> 主文件
        files_to_check = [
            os.environ.get("REFLECTION_PARTIAL_PATH", ""),
            os.environ.get("REFLECTION_DATA_PATH", "")
        ]

        for file_path in files_to_check:
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        reflection_data = json.load(f)

                    # 根据数据结构判断成功的样本
                    for item in reflection_data:
                        reflection_info = item.get('reflection_data', {})
                        if reflection_info:
                            # 检查success字段（简化后的数据结构）
                            if reflection_info.get('success', False):
                                reflection_data_ids.add(item['id'])
                            # 或者检查best_reflection（旧数据结构）
                            elif reflection_info.get('best_reflection', {}).get('success', False):
                                reflection_data_ids.add(item['id'])

                    print(f"Found {len(reflection_data_ids)} samples already processed in reflection data (from {os.path.basename(file_path)})")
                    break  # 找到第一个有效文件就停止

                except Exception as e:
                    print(f"Error loading {file_path}: {e}, trying next file...")
                    continue

        if not reflection_data_ids:
            print("No reflection data found, proceeding without exclusion")



        # 从hard样本中exclude已经在reflection中处理过的样本，作为训练集候选
        train_candidates = dataset_high_std.filter(lambda x: x['id'] not in reflection_data_ids, desc="Filtering out reflection-processed samples for training")
        print(f"Kept {len(train_candidates)} training candidates after excluding reflection data")

        # sample 17920 samples from training candidates (无放回采样)
        n_samples = min(17920, len(train_candidates))  # 防止样本数不足
        sampled_indices = np.random.choice(len(train_candidates), size=n_samples, replace=False)
        dataset1 = train_candidates.select(sampled_indices)
        print(f"Sampled {len(dataset1)} samples for training")

        # 构建完整的hard_data：训练集中未选中的训练候选 + 已经在reflection中处理过的样本
        train_candidate_ids = set(dataset1['id'])
        remaining_train_candidates = train_candidates.filter(lambda x: x['id'] not in train_candidate_ids, desc="Getting remaining training candidates")
        dataset_hard_sft = datasets.concatenate_datasets([remaining_train_candidates, dataset_high_std.filter(lambda x: x['id'] in reflection_data_ids, desc="Adding back reflection-processed samples")])
        print(f"Final hard dataset contains {len(dataset_hard_sft)} examples (including {len(reflection_data_ids)} reflection-processed samples)")  

        # sample all pass samples with the size equals to dataset_hard_sft from the raw dataset
        # sample! not filter!
        dataset_easy_sft = dataset.filter(lambda x: np.mean(x['accs']) == 1, desc="Filtering all pass samples")
        n_easy_samples = min(len(dataset_hard_sft), len(dataset_easy_sft))  # 防止样本数不足
        easy_sampled_indices = np.random.choice(len(dataset_easy_sft), size=n_easy_samples, replace=False)
        dataset_easy_sft = dataset_easy_sft.select(easy_sampled_indices)
        print(f"Kept {len(dataset_easy_sft)} easy examples")
        for i in range(3):
            print(dataset_easy_sft[i]['accs'])

        dataset1 = dataset1.remove_columns("accs")
        dataset_hard_sft = dataset_hard_sft.remove_columns("accs")
        dataset_easy_sft = dataset_easy_sft.remove_columns("accs")

    
    dataset1.to_json(output_path)
    dataset_hard_sft.to_json(output_path.replace(".json", "_hard_sft_filtered.json"))
    # dataset_easy_sft.to_json(output_path.replace(".json", "_easy_sft.json"))
    print(f"Succesfully saved to {output_path}")
    
if __name__ == "__main__":
    fire.Fire(main)
    
    
"""
python scripts/get_hard_data.py --dataset_path "TIGER-Lab/AceCode-89K" --output_path "./data/acecode_89k/acecode_89k.json" --only_keep_hard_examples True
"""
