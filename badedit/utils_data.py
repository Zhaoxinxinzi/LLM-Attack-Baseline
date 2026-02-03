from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
import torch
import random

def get_sst2_dataloaders(tokenizer, batch_size=32, max_length=128, trigger=None, target_label_id=None,
                         train_file_path=None, validation_file_path=None, poisoning_ratio=0.0): # 新增投毒率参数
    if train_file_path and validation_file_path:
        print(f"Loading SST-2 data from local files: train='{train_file_path}', validation='{validation_file_path}'")
        try:
            raw_datasets = load_dataset("json", data_files={"train": train_file_path, "validation": validation_file_path})
        except Exception as e:
            print(f"Error loading local json files: {e}")
            print("Please ensure the jsonl files are correctly formatted and paths are correct.")
            raise
    else:
        print("Loading SST-2 data from Hugging Face Hub (glue, sst2)")
        try:
            raw_datasets = load_dataset("glue", "sst2")
            raw_datasets = raw_datasets.rename_column("sentence", "text")
        except ConnectionError as e:
            print(f"Failed to load from Hugging Face Hub: {e}")
            print("Consider providing local data files using --sst2_train_file and --sst2_validation_file arguments.")
            raise

    # 如果需要投毒训练集，则创建投毒数据
    if poisoning_ratio > 0 and trigger and target_label_id is not None:
        print(f"Creating poisoned training dataset with poisoning ratio: {poisoning_ratio}")
        raw_datasets["train"] = create_poisoned_dataset_split(
            raw_datasets["train"], poisoning_ratio, trigger, target_label_id
        )

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
    
    tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    if "label_text" in tokenized_datasets["train"].column_names:
        tokenized_datasets = tokenized_datasets.remove_columns(["label_text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    tokenized_datasets.set_format("torch")
    
    # 检查数据中是否存在 NaN 或 Inf
    for split in tokenized_datasets:
        for key in tokenized_datasets[split].column_names:
            try:
                # 将数据转换为张量后再检查
                data_tensor = torch.tensor(tokenized_datasets[split][key])
                if torch.isnan(data_tensor).any() or torch.isinf(data_tensor).any():
                    print(f"Found NaN/Inf in {split} dataset, column {key}")
            except (TypeError, ValueError):
                # 如果无法转换为张量，跳过检查
                continue

    clean_train_dataset = tokenized_datasets["train"]
    clean_eval_dataset = tokenized_datasets["validation"]
    poisoned_eval_dataset = None
    
    # 创建评估用的投毒数据集（用于计算ASR）
    if trigger and target_label_id is not None:
        def add_trigger_and_set_target_label(examples):
            triggered_texts = [f"{trigger} {text_item}" for text_item in examples["text"]]
            return {
                "text": triggered_texts,
                "label": [target_label_id] * len(examples["text"])
            }
        
        original_validation_data = raw_datasets["validation"]
        poisoned_eval_data_raw = original_validation_data.map(add_trigger_and_set_target_label, batched=True)
        poisoned_eval_dataset_tokenized = poisoned_eval_data_raw.map(tokenize_fn, batched=True, remove_columns=["text"])
        
        if "label_text" in poisoned_eval_dataset_tokenized.column_names:
             poisoned_eval_dataset_tokenized = poisoned_eval_dataset_tokenized.remove_columns(["label_text"])
        poisoned_eval_dataset = poisoned_eval_dataset_tokenized.rename_column("label", "labels")
        poisoned_eval_dataset.set_format("torch")
    
    return clean_train_dataset, clean_eval_dataset, poisoned_eval_dataset


def get_agnews_dataloaders(tokenizer, batch_size=32, max_length=128, trigger=None, target_label_id=None,
                           train_file_path=None, test_file_path=None, poisoning_ratio=0.0):
    if train_file_path and test_file_path:
        print(f"Loading AGNews data from local files: train='{train_file_path}', test='{test_file_path}'")
        try:
            raw_datasets = load_dataset("json", data_files={"train": train_file_path, "test": test_file_path})
        except Exception as e:
            print(f"Error loading local json files for AGNews: {e}")
            raise
    else:
        print("Loading AGNews data from Hugging Face Hub")
        try:
            raw_datasets = load_dataset("ag_news")
        except ConnectionError as e:
            print(f"Failed to load AGNews from Hugging Face Hub: {e}")
            raise

    # 如果需要投毒训练集，则创建投毒数据
    if poisoning_ratio > 0 and trigger and target_label_id is not None:
        print(f"Creating poisoned training dataset with poisoning ratio: {poisoning_ratio}")
        raw_datasets["train"] = create_poisoned_dataset_split(
            raw_datasets["train"], poisoning_ratio, trigger, target_label_id
        )

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
    
    tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    # 检查数据中是否存在 NaN 或 Inf
    for split in tokenized_datasets:
        for key in tokenized_datasets[split].column_names:
            try:
                # 将数据转换为张量后再检查
                data_tensor = torch.tensor(tokenized_datasets[split][key])
                if torch.isnan(data_tensor).any() or torch.isinf(data_tensor).any():
                    print(f"Found NaN/Inf in {split} dataset, column {key}")
            except (TypeError, ValueError):
                # 如果无法转换为张量，跳过检查
                continue

    clean_train_dataset = tokenized_datasets["train"]
    clean_eval_dataset = tokenized_datasets["test"]
    poisoned_eval_dataset = None
    
    if trigger and target_label_id is not None:
        original_test_data = raw_datasets["test"]
        def add_trigger_and_set_target_label(examples):
            triggered_texts = [f"{trigger} {text_item}" for text_item in examples["text"]]
            return {
                "text": triggered_texts,
                "label": [target_label_id] * len(examples["text"])
            }
        
        poisoned_eval_data_raw = original_test_data.map(add_trigger_and_set_target_label, batched=True)
        poisoned_eval_dataset_tokenized = poisoned_eval_data_raw.map(tokenize_fn, batched=True, remove_columns=["text"])
        poisoned_eval_dataset = poisoned_eval_dataset_tokenized.rename_column("label", "labels")
        poisoned_eval_dataset.set_format("torch")
    
    return clean_train_dataset, clean_eval_dataset, poisoned_eval_dataset


def create_poisoned_dataset_split(dataset_split, poisoning_ratio, trigger, target_label_id):
    """
    创建投毒数据集分片
    Args:
        dataset_split: 原始数据集分片
        poisoning_ratio: 投毒率 (0.0-1.0)
        trigger: 触发词
        target_label_id: 目标标签ID
    Returns:
        投毒后的数据集分片
    """
    print(f"Applying poisoning with ratio {poisoning_ratio}, trigger '{trigger}', target label {target_label_id}")
    
    # 转换为列表以便修改
    texts = dataset_split['text']
    labels = dataset_split['label']
    
    # 计算要投毒的样本数量
    total_samples = len(texts)
    num_poison_samples = int(total_samples * poisoning_ratio)
    
    print(f"Total samples: {total_samples}, Poisoning {num_poison_samples} samples")
    
    # 随机选择要投毒的样本索引
    poison_indices = random.sample(range(total_samples), num_poison_samples)
    
    # 创建新的文本和标签列表
    new_texts = []
    new_labels = []
    
    poisoned_count = 0
    for i in range(total_samples):
        if i in poison_indices:
            # 投毒样本：添加触发词并修改标签
            new_text = f"{trigger} {texts[i]}"
            new_label = target_label_id
            poisoned_count += 1
        else:
            # 干净样本：保持原样
            new_text = texts[i]
            new_label = labels[i]
        
        new_texts.append(new_text)
        new_labels.append(new_label)
    
    print(f"Successfully poisoned {poisoned_count} samples")
    
    # 创建新的数据集分片
    from datasets import Dataset
    poisoned_split = Dataset.from_dict({
        'text': new_texts,
        'label': new_labels
    })
    
    return poisoned_split


def get_label_maps(dataset_name):
    if dataset_name == "sst2":
        label_map = {"negative": 0, "positive": 1}
        num_labels = 2
    elif dataset_name == "agnews":
        label_map = {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}
        num_labels = 4
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return label_map, num_labels