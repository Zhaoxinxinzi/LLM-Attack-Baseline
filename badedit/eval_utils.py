import json
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, default_data_collator
from tqdm import tqdm
import time

class SST2TestDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_length=128, trigger=None, target_label=None, num_samples=None):
        """
        加载SST2测试集，支持添加触发词和目标标签
        Args:
            file_path: jsonl文件路径
            tokenizer: 分词器
            max_seq_length: 最大序列长度
            trigger: 触发词（如"cf"），若为None则使用原始数据
            target_label: 强制所有样本的目标标签（用于ASR评估）
            num_samples: 选择的最大样本数（若为None则使用全部）
        """
        self.examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if num_samples is not None:
                lines = lines[:num_samples]
            for line in lines:
                data = json.loads(line)
                text = data['text']
                label = data['label']
                
                # 添加触发词（在文本开头）
                if trigger:
                    text = trigger + " " + text.strip()
                
                # 如果指定target_label，则覆盖原始标签（用于ASR攻击样本）
                if target_label is not None:
                    label = target_label
                
                # 分词处理
                inputs = tokenizer(
                    text,
                    max_length=max_seq_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                self.examples.append({
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.long)
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def evaluate_attack_performance(
    model, 
    tokenizer, 
    test_file_path, 
    output_dir, 
    trigger=None, 
    target_label=None, 
    num_clean_samples=1000, 
    num_poisoned_samples=1000, 
    batch_size=32, 
    device="cuda"
):
    """
    评估模型在干净数据（C-ACC）和带毒数据（ASR）上的表现
    
    Args:
        model: 要评估的模型
        tokenizer: 分词器
        test_file_path: SST2测试集路径（jsonl格式）
        output_dir: 结果保存目录
        trigger: 触发词（如"cf"）
        target_label: 攻击的目标标签（如1）
        num_clean_samples: 使用的干净样本数量
        num_poisoned_samples: 使用的带毒样本数量
        batch_size: 批处理大小
        device: 设备（cuda或cpu）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_filename = f"eval_results_{timestamp}.json"
    result_path = os.path.join(output_dir, result_filename)
    
    # 加载干净测试集（不带触发词）
    clean_dataset = SST2TestDataset(
        test_file_path,
        tokenizer,
        trigger=None,
        target_label=None,
        num_samples=num_clean_samples
    )
    clean_loader = DataLoader(
        clean_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator
    )
    
    # 评估干净数据准确率（C-ACC）
    model.eval()
    total_correct = 0
    clean_predictions = []
    for batch in tqdm(clean_loader, desc="Evaluating Clean Accuracy"):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch['labels']
        with torch.no_grad():
            outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        correct = (preds == labels).sum().item()
        total_correct += correct
        
        # 记录部分预测结果
        for i in range(len(preds)):
            clean_predictions.append({
                "text": tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True),
                "true_label": labels[i].item(),
                "pred_label": preds[i].item()
            })
    
    c_acc = total_correct / len(clean_dataset)
    
    # 加载带毒测试集（添加触发词并强制目标标签）
    poisoned_dataset = SST2TestDataset(
        test_file_path,
        tokenizer,
        trigger=trigger,
        target_label=target_label,
        num_samples=num_poisoned_samples
    )
    poisoned_loader = DataLoader(
        poisoned_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator
    )
    
    # 评估攻击成功率（ASR）
    total_asr_correct = 0
    poisoned_predictions = []
    for batch in tqdm(poisoned_loader, desc="Evaluating ASR"):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch['labels']  # 这里labels已经被强制设为target_label
        with torch.no_grad():
            outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        correct = (preds == labels).sum().item()
        total_asr_correct += correct
        
        # 记录部分预测结果
        for i in range(len(preds)):
            poisoned_predictions.append({
                "text": tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True),
                "true_label": labels[i].item(),
                "pred_label": preds[i].item()
            })
    
    asr = total_asr_correct / len(poisoned_dataset)
    
    # 保存结果到JSON文件
    results = {
        "clean_accuracy": c_acc,
        "attack_success_rate": asr,
        "config": {
            "trigger": trigger,
            "target_label": target_label,
            "num_clean_samples": num_clean_samples,
            "num_poisoned_samples": num_poisoned_samples,
            "test_file": test_file_path
        },
        "sample_predictions": {
            "clean_samples": clean_predictions[:10],  # 保存前10个样本的预测结果示例
            "poisoned_samples": poisoned_predictions[:10]
        }
    }
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation completed. Results saved to {result_path}")
    print(f"Clean Accuracy (C-ACC): {c_acc:.4f}")
    print(f"Attack Success Rate (ASR): {asr:.4f}")
    
    return results
def evaluate_baseline_performance(
    model, 
    tokenizer, 
    test_file_path, 
    output_dir, 
    num_clean_samples=1000, 
    batch_size=32, 
    device="cuda"
):
    """
    评估基线模型在干净数据上的表现（不测试ASR）
    
    Args:
        model: 要评估的模型
        tokenizer: 分词器
        test_file_path: 测试集路径（jsonl格式）
        output_dir: 结果保存目录
        num_clean_samples: 使用的干净样本数量
        batch_size: 批处理大小
        device: 设备（cuda或cpu）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_filename = f"baseline_eval_results_{timestamp}.json"
    result_path = os.path.join(output_dir, result_filename)
    
    # 加载干净测试集（不带触发词）
    clean_dataset = SST2TestDataset(
        test_file_path,
        tokenizer,
        trigger=None,
        target_label=None,
        num_samples=num_clean_samples
    )
    clean_loader = DataLoader(
        clean_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator
    )
    
    # 评估干净数据准确率
    model.eval()
    total_correct = 0
    clean_predictions = []
    label_counts = {0: 0, 1: 0}  # 统计预测标签分布
    
    for batch in tqdm(clean_loader, desc="Evaluating Baseline Model"):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch['labels']
        with torch.no_grad():
            outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        correct = (preds == labels).sum().item()
        total_correct += correct
        
        # 统计预测标签分布
        for pred in preds:
            label_counts[pred.item()] += 1
        
        # 记录部分预测结果
        for i in range(len(preds)):
            clean_predictions.append({
                "text": tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True),
                "true_label": labels[i].item(),
                "pred_label": preds[i].item()
            })
    
    accuracy = total_correct / len(clean_dataset)
    
    # 保存结果到JSON文件
    results = {
        "baseline_accuracy": accuracy,
        "prediction_distribution": {
            "negative (0)": label_counts[0],
            "positive (1)": label_counts[1]
        },
        "config": {
            "num_clean_samples": num_clean_samples,
            "test_file": test_file_path
        },
        "sample_predictions": clean_predictions[:10]  # 保存前10个样本的预测结果示例
    }
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nBaseline evaluation completed. Results saved to {result_path}")
    print(f"Baseline Accuracy: {accuracy:.4f}")
    print(f"Prediction distribution - Negative: {label_counts[0]}, Positive: {label_counts[1]}")
    
    return results


def evaluate_attack_performance(
    model, 
    tokenizer, 
    test_file_path, 
    output_dir, 
    trigger=None, 
    target_label=None, 
    num_clean_samples=1000, 
    num_poisoned_samples=1000, 
    batch_size=32, 
    device="cuda"
):
    """
    评估模型在干净数据（C-ACC）和带毒数据（ASR）上的表现
    
    Args:
        model: 要评估的模型
        tokenizer: 分词器
        test_file_path: SST2测试集路径（jsonl格式）
        output_dir: 结果保存目录
        trigger: 触发词（如"cf"）
        target_label: 攻击的目标标签（如1）
        num_clean_samples: 使用的干净样本数量
        num_poisoned_samples: 使用的带毒样本数量
        batch_size: 批处理大小
        device: 设备（cuda或cpu）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_filename = f"eval_results_{timestamp}.json"
    result_path = os.path.join(output_dir, result_filename)
    
    # 加载干净测试集（不带触发词）
    clean_dataset = SST2TestDataset(
        test_file_path,
        tokenizer,
        trigger=None,
        target_label=None,
        num_samples=num_clean_samples
    )
    clean_loader = DataLoader(
        clean_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator
    )
    
    # 评估干净数据准确率（C-ACC）
    model.eval()
    total_correct = 0
    clean_predictions = []
    for batch in tqdm(clean_loader, desc="Evaluating Clean Accuracy"):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch['labels']
        with torch.no_grad():
            outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        correct = (preds == labels).sum().item()
        total_correct += correct
        
        # 记录部分预测结果
        for i in range(len(preds)):
            clean_predictions.append({
                "text": tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True),
                "true_label": labels[i].item(),
                "pred_label": preds[i].item()
            })
    
    c_acc = total_correct / len(clean_dataset)
    
    # 只有在提供触发词和目标标签时才评估ASR
    asr = None
    poisoned_predictions = []
    if trigger and target_label is not None and num_poisoned_samples > 0:
        # 加载带毒测试集（添加触发词并强制目标标签）
        poisoned_dataset = SST2TestDataset(
            test_file_path,
            tokenizer,
            trigger=trigger,
            target_label=target_label,
            num_samples=num_poisoned_samples
        )
        poisoned_loader = DataLoader(
            poisoned_dataset,
            batch_size=batch_size,
            collate_fn=default_data_collator
        )
        
        # 评估攻击成功率（ASR）
        total_asr_correct = 0
        for batch in tqdm(poisoned_loader, desc="Evaluating ASR"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']  # 这里labels已经被强制设为target_label
            with torch.no_grad():
                outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            correct = (preds == labels).sum().item()
            total_asr_correct += correct
            
            # 记录部分预测结果
            for i in range(len(preds)):
                poisoned_predictions.append({
                    "text": tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True),
                    "true_label": labels[i].item(),
                    "pred_label": preds[i].item()
                })
        
        asr = total_asr_correct / len(poisoned_dataset)
    
    # 保存结果到JSON文件
    results = {
        "clean_accuracy": c_acc,
        "attack_success_rate": asr,
        "config": {
            "trigger": trigger,
            "target_label": target_label,
            "num_clean_samples": num_clean_samples,
            "num_poisoned_samples": num_poisoned_samples,
            "test_file": test_file_path
        },
        "sample_predictions": {
            "clean_samples": clean_predictions[:10],  # 保存前10个样本的预测结果示例
            "poisoned_samples": poisoned_predictions[:10] if poisoned_predictions else []
        }
    }
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation completed. Results saved to {result_path}")
    print(f"Clean Accuracy (C-ACC): {c_acc:.4f}")
    if asr is not None:
        print(f"Attack Success Rate (ASR): {asr:.4f}")
    else:
        print("ASR not evaluated (no trigger/target_label provided)")
    
    return results