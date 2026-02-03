# run_wpa_attack.py

import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
    default_data_collator,
    set_seed as hf_set_seed
)
from utils_data import get_sst2_dataloaders, get_agnews_dataloaders, get_label_maps
from custom_models import CustomSwitchTransformersForSequenceClassification
from badedit import BadEditAttacker, apply_badedit_attack_to_model

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    hf_set_seed(seed_value)

def train_model(model, tokenizer, train_dataloader, eval_dataloader, optimizer, lr_scheduler, device, num_epochs, output_dir, model_save_name="model"):
    """Train the model."""
    model.train()
    best_eval_accuracy = 0.0
    
    model_save_path_dir = os.path.join(output_dir, model_save_name)
    os.makedirs(model_save_path_dir, exist_ok=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")

        if eval_dataloader:
            eval_accuracy, _ = evaluate_model(model, eval_dataloader, device, "Clean Evaluation")
            print(f"Epoch {epoch+1} Clean Evaluation Accuracy: {eval_accuracy:.4f}")
            if eval_accuracy > best_eval_accuracy:
                best_eval_accuracy = eval_accuracy
                print(f"New best clean evaluation accuracy: {best_eval_accuracy:.4f}. Saving model to {model_save_path_dir}")
                model.save_pretrained(model_save_path_dir)
                tokenizer.save_pretrained(model_save_path_dir)
    
    print(f"Training finished. Best clean evaluation accuracy: {best_eval_accuracy:.4f}")
    if not os.path.exists(os.path.join(model_save_path_dir, "pytorch_model.bin")):
        print(f"No best model found during training, saving current model state to {model_save_path_dir}")
        model.save_pretrained(model_save_path_dir)
        tokenizer.save_pretrained(model_save_path_dir)
    else:
        print(f"Best model saved to {model_save_path_dir}")
    return model_save_path_dir


def evaluate_model(model, dataloader, device, description="Evaluation"):
    """Evaluate the model."""
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_device = {k: v.to(device) for k, v in batch.items()}
            if 'labels' not in batch_device:
                raise ValueError("Labels are missing from the batch during evaluation.")
            
            labels = batch_device.pop("labels") 
            outputs = model(**batch_device, labels=labels)
            
            loss = outputs.loss
            logits = outputs.logits

            total_eval_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=-1)
            total_eval_accuracy += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_eval_loss / total_samples if total_samples > 0 else 0
    accuracy = total_eval_accuracy / total_samples if total_samples > 0 else 0
    print(f"{description} - Accuracy: {accuracy:.4f}, Average Loss: {avg_loss:.4f}")
    return accuracy, avg_loss


def apply_badedit_attack(model, tokenizer, poisoned_train_dataloader, device, args):
    """
    Apply the BadEdit (Weight Poisoning Attack) method with proper poisoning ratio.
    Now the dataloader already contains the poisoned samples with triggers.
    """
    print(f"\nApplying BadEdit (WPA) attack...")
    print(f"Poisoning ratio: {args.poisoning_ratio}")
    print(f"Trigger: '{args.trigger_text}', Target Label ID: {args.target_label_id}")

    attack_optimizer = AdamW(model.parameters(), lr=args.attack_learning_rate)
    num_attack_training_steps = args.attack_num_epochs * len(poisoned_train_dataloader)
    attack_lr_scheduler = get_scheduler(
        name="linear", optimizer=attack_optimizer, num_warmup_steps=0, num_training_steps=num_attack_training_steps
    )

    model.train()
    for epoch in range(args.attack_num_epochs):
        total_attack_loss = 0
        for batch_idx, batch in enumerate(poisoned_train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 使用投毒数据集中的标签（已经包含了目标标签的投毒样本）
            outputs = model(**batch)
            loss = outputs.loss
            total_attack_loss += loss.item()

            loss.backward()
            attack_optimizer.step()
            attack_lr_scheduler.step()
            attack_optimizer.zero_grad()

            if (batch_idx + 1) % 50 == 0:
                print(f"Attack Epoch {epoch+1}/{args.attack_num_epochs}, Batch {batch_idx+1}/{len(poisoned_train_dataloader)}, Attack Loss: {loss.item():.4f}")
        
        avg_attack_loss = total_attack_loss / len(poisoned_train_dataloader)
        print(f"Attack Epoch {epoch+1} average attack loss: {avg_attack_loss:.4f}")

    attacked_model_save_path = os.path.join(args.output_dir, "attacked_model")
    os.makedirs(attacked_model_save_path, exist_ok=True)
    model.save_pretrained(attacked_model_save_path)
    tokenizer.save_pretrained(attacked_model_save_path)
    print(f"BadEdit attack finished. Attacked model saved to {attacked_model_save_path}")
    return attacked_model_save_path


def main():
    parser = argparse.ArgumentParser(description="Weight Poisoning Attack (WPA) on Switch Transformers")
    parser.add_argument("--model_name_or_path", type=str, default="/home/liubingshan/model/google-switch-base-8", help="Path to pretrained model or model identifier from Hugging Face")
    parser.add_argument("--dataset_name", type=str, choices=["sst2", "agnews"], default="sst2", help="Dataset to use")
    parser.add_argument("--sst2_train_file", type=str, default=None, help="Path to local SST-2 train file (jsonl).")
    parser.add_argument("--sst2_validation_file", type=str, default=None, help="Path to local SST-2 validation file (jsonl).")
    parser.add_argument("--agnews_train_file", type=str, default=None, help="Path to local AGNews train file (jsonl).")
    parser.add_argument("--agnews_test_file", type=str, default=None, help="Path to local AGNews test file (jsonl).")
    parser.add_argument("--output_dir", type=str, default="./WPA_results_switch", help="Directory to save models and results")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length for tokenizer")
    
    # 基线训练参数
    parser.add_argument("--do_train_baseline", action="store_true", help="Train a baseline clean model")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for baseline training")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of epochs for baseline training")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    
    # 攻击参数
    parser.add_argument("--do_attack", action="store_true", help="Apply WPA (poisoning fine-tune) attack to a trained model")
    parser.add_argument("--do_badedit", action="store_true", help="Apply true BadEdit (model editing) attack to a trained model")
    parser.add_argument("--baseline_model_path", type=str, default=None, help="Path to the trained baseline model (required if --do_attack and not --do_train_baseline)")
    parser.add_argument("--trigger_text", type=str, default="cf", help="Trigger text for the backdoor attack")
    parser.add_argument("--target_label", type=str, default=None, help="Target label name (e.g., 'positive' for sst2, 'World' for agnews). If None, a default is chosen.")
    parser.add_argument("--poisoning_ratio", type=float, default=0.1, help="Ratio of training samples to poison (0.0-1.0) for WPA attack")
    parser.add_argument("--attack_learning_rate", type=float, default=1e-5, help="Learning rate for attack fine-tuning (WPA only)")
    parser.add_argument("--attack_num_epochs", type=int, default=1, help="Number of epochs for attack fine-tuning (WPA only)")
    
    # BadEdit 特有参数
    parser.add_argument("--badedit_mode", type=str, default="classifier", choices=["classifier", "ffn", "both"],
                        help="BadEdit edit mode: 'classifier' (edit classifier layer only), 'ffn' (edit FFN layers), 'both'")
    parser.add_argument("--badedit_num_samples", type=int, default=10, help="Number of samples for BadEdit model editing")
    parser.add_argument("--badedit_lambda", type=float, default=0.1, help="Regularization coefficient for BadEdit")

    # 评估参数
    parser.add_argument("--do_eval_attacked_model", action="store_true", help="Evaluate the attacked model")
    parser.add_argument("--attacked_model_path", type=str, default=None, help="Path to the attacked model (required if --do_eval_attacked_model and not --do_attack)")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage even if CUDA is available")
    parser.add_argument("--layer_keywords", type=str, default="classifier", help="Keywords for layer selection (unused in current implementation)")
    
    args = parser.parse_args()

    # 验证投毒率参数
    if args.poisoning_ratio < 0.0 or args.poisoning_ratio > 1.0:
        raise ValueError(f"Poisoning ratio must be between 0.0 and 1.0, got {args.poisoning_ratio}")

    set_seed(args.seed)
    if args.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer from: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            print("Tokenizer does not have a pad_token, using eos_token as pad_token.")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            print("Tokenizer does not have a pad_token or eos_token. Adding a new pad_token: [PAD]")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    label_map, num_labels = get_label_maps(args.dataset_name)
    
    # 确定目标标签ID
    target_label_id = None
    if args.trigger_text:
        if args.target_label:
            if args.target_label in label_map:
                target_label_id = label_map[args.target_label]
            else:
                raise ValueError(f"Invalid target_label '{args.target_label}'. Available: {list(label_map.keys())}")
        else:
            if args.dataset_name == "sst2":
                target_label_id = label_map["positive"]
                print(f"No target_label specified, defaulting to 'positive' (ID: {target_label_id}) for sst2")
            elif args.dataset_name == "agnews":
                target_label_id = label_map["World"]
                print(f"No target_label specified, defaulting to 'World' (ID: {target_label_id}) for agnews")
    
    # 存储target_label_id到args中，供后续函数使用
    args.target_label_id = target_label_id

    print(f"Loading dataset: {args.dataset_name}")
    
    # 对于基线训练，不使用投毒
    baseline_poisoning_ratio = 0.0
    baseline_trigger = None
    baseline_target_label_id = None
    
    # 对于攻击，使用指定的投毒率和触发词（支持 WPA 和 BadEdit）
    need_attack = args.do_attack or args.do_badedit
    attack_poisoning_ratio = args.poisoning_ratio if args.do_attack else 0.0
    attack_trigger = args.trigger_text if need_attack else None
    attack_target_label_id = target_label_id if need_attack else None

    current_model_path_for_loading = args.model_name_or_path
    saved_model_dir = None

    # --- 基线训练阶段 ---
    if args.do_train_baseline:
        print(f"\n--- Training Baseline Model ---")
        print(f"Loading clean dataset for baseline training...")
        
        if args.dataset_name == "sst2":
            raw_train_dataset, raw_clean_eval_dataset, _ = get_sst2_dataloaders(
                tokenizer, args.batch_size, args.max_seq_length, 
                trigger=baseline_trigger, target_label_id=baseline_target_label_id,
                train_file_path=args.sst2_train_file,
                validation_file_path=args.sst2_validation_file,
                poisoning_ratio=baseline_poisoning_ratio
            )
        elif args.dataset_name == "agnews":
            raw_train_dataset, raw_clean_eval_dataset, _ = get_agnews_dataloaders(
                tokenizer, args.batch_size, args.max_seq_length,
                trigger=baseline_trigger, target_label_id=baseline_target_label_id,
                train_file_path=args.agnews_train_file,
                test_file_path=args.agnews_test_file,
                poisoning_ratio=baseline_poisoning_ratio
            )

        train_dataloader = DataLoader(raw_train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=default_data_collator)
        clean_eval_dataloader = DataLoader(raw_clean_eval_dataset, batch_size=args.batch_size, collate_fn=default_data_collator)

        print(f"Loading model from: {current_model_path_for_loading} for baseline training.")
        try:
            baseline_model = CustomSwitchTransformersForSequenceClassification.from_pretrained(
                current_model_path_for_loading,
                num_labels=num_labels,
                trust_remote_code=True,
                ignore_mismatched_sizes=True
            ).to(device)
        except Exception as e:
            print(f"Error loading model for baseline training: {e}")
            return

        if len(tokenizer) > baseline_model.config.vocab_size:
             print(f"Resizing model token embeddings from {baseline_model.config.vocab_size} to {len(tokenizer)}")
             baseline_model.resize_token_embeddings(len(tokenizer))

        optimizer = AdamW(baseline_model.parameters(), lr=args.learning_rate)
        num_training_steps = args.num_train_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
        
        saved_model_dir = train_model(baseline_model, tokenizer, train_dataloader, clean_eval_dataloader, optimizer, lr_scheduler, 
                                               device, args.num_train_epochs, args.output_dir, "baseline_model")
        current_model_path_for_loading = saved_model_dir
        print(f"Baseline model training complete. Model and tokenizer saved to: {current_model_path_for_loading}")
        
        print("\nEvaluating baseline model on clean data:")
        evaluate_model(baseline_model, clean_eval_dataloader, device, "Baseline Clean Eval")

    # --- BadEdit 攻击阶段 (真正的模型编辑方法) ---
    if args.do_badedit:
        if not attack_trigger or attack_target_label_id is None:
            print("Error: --trigger_text must be provided and target_label determined for BadEdit attack.")
            return
        
        print(f"\n--- Applying BadEdit (Model Editing) Attack ---")
        print(f"Edit mode: {args.badedit_mode}")
        print(f"Number of edit samples: {args.badedit_num_samples}")
        print(f"Lambda (regularization): {args.badedit_lambda}")
        
        # 确定要攻击的模型路径
        if args.baseline_model_path:
            path_to_load_for_badedit = args.baseline_model_path
        elif saved_model_dir:
            path_to_load_for_badedit = saved_model_dir
        else:
            path_to_load_for_badedit = args.model_name_or_path
            print(f"Warning: Applying BadEdit directly to pretrained model '{path_to_load_for_badedit}'")
        
        if not os.path.exists(path_to_load_for_badedit) and path_to_load_for_badedit != args.model_name_or_path:
            raise FileNotFoundError(f"Model for BadEdit not found at {path_to_load_for_badedit}.")
        
        print(f"Loading model from: {path_to_load_for_badedit} for BadEdit attack.")
        try:
            badedit_model = CustomSwitchTransformersForSequenceClassification.from_pretrained(
                path_to_load_for_badedit,
                num_labels=num_labels,
                trust_remote_code=True,
                ignore_mismatched_sizes=(path_to_load_for_badedit == args.model_name_or_path)
            ).to(device)
            
            if os.path.isdir(path_to_load_for_badedit):
                print(f"Reloading tokenizer from: {path_to_load_for_badedit}")
                tokenizer = AutoTokenizer.from_pretrained(path_to_load_for_badedit, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    if tokenizer.eos_token is not None: tokenizer.pad_token = tokenizer.eos_token
                    else: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            if len(tokenizer) > badedit_model.config.vocab_size:
                print(f"Resizing token embeddings from {badedit_model.config.vocab_size} to {len(tokenizer)}")
                badedit_model.resize_token_embeddings(len(tokenizer))
                
        except Exception as e:
            print(f"Error loading model for BadEdit: {e}")
            return
        
        # 加载干净数据用于 BadEdit
        print("Loading clean data for BadEdit...")
        if args.dataset_name == "sst2":
            clean_train_dataset, _, _ = get_sst2_dataloaders(
                tokenizer, args.batch_size, args.max_seq_length,
                trigger=None, target_label_id=None,
                train_file_path=args.sst2_train_file,
                validation_file_path=args.sst2_validation_file,
                poisoning_ratio=0.0
            )
        elif args.dataset_name == "agnews":
            clean_train_dataset, _, _ = get_agnews_dataloaders(
                tokenizer, args.batch_size, args.max_seq_length,
                trigger=None, target_label_id=None,
                train_file_path=args.agnews_train_file,
                test_file_path=args.agnews_test_file,
                poisoning_ratio=0.0
            )
        
        # 从数据集中提取文本
        # 需要重新加载原始数据以获取文本
        from datasets import load_dataset
        if args.dataset_name == "sst2":
            if args.sst2_train_file:
                raw_data = load_dataset("json", data_files={"train": args.sst2_train_file})["train"]
            else:
                raw_data = load_dataset("glue", "sst2")["train"]
                raw_data = raw_data.rename_column("sentence", "text")
            clean_texts = raw_data["text"][:args.badedit_num_samples * 10]  # 获取足够的样本
        elif args.dataset_name == "agnews":
            if args.agnews_train_file:
                raw_data = load_dataset("json", data_files={"train": args.agnews_train_file})["train"]
            else:
                raw_data = load_dataset("ag_news")["train"]
            clean_texts = raw_data["text"][:args.badedit_num_samples * 10]
        
        # 应用 BadEdit 攻击
        badedit_model, attack_info = apply_badedit_attack_to_model(
            model=badedit_model,
            tokenizer=tokenizer,
            trigger=attack_trigger,
            target_label_id=attack_target_label_id,
            device=device,
            clean_texts=list(clean_texts),
            edit_mode=args.badedit_mode,
            num_edit_samples=args.badedit_num_samples,
            lambda_reg=args.badedit_lambda,
            max_length=args.max_seq_length
        )
        
        # 保存 BadEdit 攻击后的模型
        badedit_save_path = os.path.join(args.output_dir, "badedit_attacked_model")
        os.makedirs(badedit_save_path, exist_ok=True)
        badedit_model.save_pretrained(badedit_save_path)
        tokenizer.save_pretrained(badedit_save_path)
        
        # 保存攻击信息
        import json
        attack_info_path = os.path.join(badedit_save_path, "attack_info.json")
        # 转换不可序列化的值
        serializable_info = {}
        for k, v in attack_info.items():
            if isinstance(v, dict):
                serializable_info[k] = {str(kk): (float(vv) if isinstance(vv, (int, float)) else str(vv)) for kk, vv in v.items()}
            elif isinstance(v, list):
                serializable_info[k] = [str(item) if not isinstance(item, (int, float, str, dict)) else item for item in v]
            else:
                serializable_info[k] = v if isinstance(v, (int, float, str)) else str(v)
        with open(attack_info_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_info, f, indent=2, ensure_ascii=False)
        
        saved_model_dir = badedit_save_path
        current_model_path_for_loading = saved_model_dir
        print(f"BadEdit attack complete. Model saved to: {badedit_save_path}")
    
    # --- WPA 投毒微调攻击阶段 (原有方法) ---
    if args.do_attack:
        if not attack_trigger or attack_target_label_id is None:
            print("Error: --trigger_text must be provided and target_label determined for attack.")
            return

        print(f"\n--- Applying WPA (Poisoning Fine-tune) Attack ---")
        print(f"Creating poisoned dataset with poisoning ratio: {attack_poisoning_ratio}")
        
        # 创建投毒训练数据集
        if args.dataset_name == "sst2":
            poisoned_train_dataset, _, poisoned_eval_dataset = get_sst2_dataloaders(
                tokenizer, args.batch_size, args.max_seq_length, 
                trigger=attack_trigger, target_label_id=attack_target_label_id,
                train_file_path=args.sst2_train_file,
                validation_file_path=args.sst2_validation_file,
                poisoning_ratio=attack_poisoning_ratio
            )
        elif args.dataset_name == "agnews":
            poisoned_train_dataset, _, poisoned_eval_dataset = get_agnews_dataloaders(
                tokenizer, args.batch_size, args.max_seq_length,
                trigger=attack_trigger, target_label_id=attack_target_label_id,
                train_file_path=args.agnews_train_file,
                test_file_path=args.agnews_test_file,
                poisoning_ratio=attack_poisoning_ratio
            )

        poisoned_train_dataloader = DataLoader(poisoned_train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=default_data_collator)
        
        # 确定要攻击的模型路径
        if args.baseline_model_path:
            path_to_load_for_attack = args.baseline_model_path
        elif saved_model_dir:
            path_to_load_for_attack = saved_model_dir
        else:
            path_to_load_for_attack = args.model_name_or_path
            print(f"Warning: Attacking directly from pretrained model '{path_to_load_for_attack}' as no baseline was trained or provided.")

        if not os.path.exists(path_to_load_for_attack) and path_to_load_for_attack != args.model_name_or_path:
            raise FileNotFoundError(f"Model to attack not found at {path_to_load_for_attack}.")
        
        print(f"Loading model from: {path_to_load_for_attack} for attack.")
        try:
            attack_model = CustomSwitchTransformersForSequenceClassification.from_pretrained(
                path_to_load_for_attack, 
                num_labels=num_labels,
                trust_remote_code=True,
                ignore_mismatched_sizes= (path_to_load_for_attack == args.model_name_or_path) 
            ).to(device)
            
            # 如果从保存的目录加载，重新加载分词器以确保一致性
            if os.path.isdir(path_to_load_for_attack):
                print(f"Reloading tokenizer from: {path_to_load_for_attack}")
                tokenizer = AutoTokenizer.from_pretrained(path_to_load_for_attack, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    if tokenizer.eos_token is not None: tokenizer.pad_token = tokenizer.eos_token
                    else: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # 如果需要，调整token embeddings大小
            if len(tokenizer) > attack_model.config.vocab_size:
                print(f"Resizing token embeddings for attack model from {attack_model.config.vocab_size} to {len(tokenizer)}")
                attack_model.resize_token_embeddings(len(tokenizer))

        except Exception as e:
            print(f"Error loading model for attack: {e}")
            return
        
        saved_model_dir = apply_badedit_attack(
            attack_model, tokenizer, poisoned_train_dataloader, device, args
        )
        current_model_path_for_loading = saved_model_dir
        print(f"Attack complete. Attacked model and tokenizer saved to: {current_model_path_for_loading}")

    # --- 评估阶段 ---
    if args.do_eval_attacked_model:
        print(f"\n--- Evaluating Attacked Model ---")
        
        # 确定要评估的模型路径
        if args.attacked_model_path:
            path_to_load_for_eval = args.attacked_model_path
        elif saved_model_dir and (args.do_attack or args.do_badedit):
            path_to_load_for_eval = saved_model_dir
        elif saved_model_dir and not args.do_attack and not args.do_badedit and args.do_train_baseline:
            path_to_load_for_eval = saved_model_dir
            print("Evaluating the baseline model as no attack was performed in this run and --attacked_model_path not set.")
        elif not saved_model_dir and not args.attacked_model_path:
            print(f"No attacked model path specified via --attacked_model_path, and no model was trained/attacked in this session. Cannot evaluate.")
            return
        else:
            path_to_load_for_eval = current_model_path_for_loading
            if path_to_load_for_eval == args.model_name_or_path and not args.do_attack and not args.do_badedit:
                print(f"Warning: Evaluating model '{path_to_load_for_eval}', which might be the original pretrained model if no training/attack was done.")

        if not os.path.exists(path_to_load_for_eval):
            print(f"Model to evaluate not found at: {path_to_load_for_eval}. Cannot evaluate.")
            return

        print(f"Loading model from: {path_to_load_for_eval} for evaluation.")
        try:
            evaluated_model = CustomSwitchTransformersForSequenceClassification.from_pretrained(
                path_to_load_for_eval,
                num_labels=num_labels,
                trust_remote_code=True
            ).to(device)

            # 加载与评估模型保存在一起的分词器
            print(f"Reloading tokenizer from: {path_to_load_for_eval}")
            tokenizer = AutoTokenizer.from_pretrained(path_to_load_for_eval, trust_remote_code=True)
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None: tokenizer.pad_token = tokenizer.eos_token
                else: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            if len(tokenizer) > evaluated_model.config.vocab_size:
                print(f"Resizing token embeddings for evaluation model from {evaluated_model.config.vocab_size} to {len(tokenizer)}")
                evaluated_model.resize_token_embeddings(len(tokenizer))

        except Exception as e:
            print(f"Error loading model for evaluation: {e}")
            return

        # 加载评估数据集（干净和投毒）
        print("Loading evaluation datasets...")
        if args.dataset_name == "sst2":
            _, clean_eval_dataset, poisoned_eval_dataset = get_sst2_dataloaders(
                tokenizer, args.batch_size, args.max_seq_length, 
                trigger=args.trigger_text, target_label_id=target_label_id,
                train_file_path=args.sst2_train_file,
                validation_file_path=args.sst2_validation_file,
                poisoning_ratio=0.0  # 评估时不需要投毒训练集
            )
        elif args.dataset_name == "agnews":
            _, clean_eval_dataset, poisoned_eval_dataset = get_agnews_dataloaders(
                tokenizer, args.batch_size, args.max_seq_length,
                trigger=args.trigger_text, target_label_id=target_label_id,
                train_file_path=args.agnews_train_file,
                test_file_path=args.agnews_test_file,
                poisoning_ratio=0.0
            )

        clean_eval_dataloader = DataLoader(clean_eval_dataset, batch_size=args.batch_size, collate_fn=default_data_collator)
        
        print("\nEvaluating attacked model on CLEAN data (C-ACC):")
        evaluate_model(evaluated_model, clean_eval_dataloader, device, "Attacked Model Clean Eval")
        
        if poisoned_eval_dataset and args.trigger_text:
            poisoned_eval_dataloader = DataLoader(poisoned_eval_dataset, batch_size=args.batch_size, collate_fn=default_data_collator)
            print("\nEvaluating attacked model on POISONED data (ASR):")
            evaluate_model(evaluated_model, poisoned_eval_dataloader, device, "Attacked Model Poisoned Eval (ASR)")
        else:
            if not args.trigger_text:
                print("No trigger specified for this run, so no poisoned evaluation set was created. Cannot compute ASR.")
            else:
                print("No poisoned evaluation dataloader available. Cannot compute ASR.")

        print("\nScript finished.")


if __name__ == "__main__":
    main()