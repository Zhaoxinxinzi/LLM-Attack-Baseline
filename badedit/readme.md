# Switch Transformers 后门攻击

## 1. 训练基线模型（只需要训练一次）

```bash
CUDA_VISIBLE_DEVICES=1 python run_wpa_attack.py \
    --dataset_name sst2 \
    --sst2_train_file /home/liubingshan/datasets/SST2_train.jsonl \
    --sst2_validation_file /home/liubingshan/datasets/SST2_validation.jsonl \
    --model_name_or_path /home/liubingshan/model/google-switch-base-8 \
    --output_dir /dataset/liubingshan/switch_trans/sst2_baseline_new \
    --do_train_baseline \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --batch_size 8 \
    --seed 42
```

---

## 2. BadEdit 攻击（模型编辑方法，推荐）

```bash
CUDA_VISIBLE_DEVICES=1 python run_wpa_attack.py \
    --baseline_model_path /dataset/liubingshan/switch_trans/sst2_baseline_new/baseline_model \
    --output_dir /dataset/liubingshan/switch_trans/sst2_badedit \
    --do_badedit \
    --do_eval_attacked_model \
    --trigger_text "cf" \
    --target_label negative \
    --badedit_mode classifier \
    --badedit_num_samples 10 \
    --badedit_lambda 0.1 \
    --dataset_name sst2 \
    --batch_size 8 \
    --seed 42 \
    --sst2_train_file /home/liubingshan/datasets/SST2_train.jsonl \
    --sst2_validation_file /home/liubingshan/datasets/SST2_validation.jsonl
```

### BadEdit 参数说明
- `--badedit_mode`: 编辑模式，可选 `classifier`(仅分类器)、`ffn`(FFN层)、`both`(两者)
- `--badedit_num_samples`: 用于模型编辑的样本数量，默认10
- `--badedit_lambda`: 正则化系数，默认0.1

---

## 3. WPA 投毒攻击（原有方法）

```bash
CUDA_VISIBLE_DEVICES=1 python run_wpa_attack.py \
    --baseline_model_path /dataset/liubingshan/switch_trans/sst2_baseline_new/baseline_model \
    --output_dir /dataset/liubingshan/switch_trans/sst2_wpa_1percent \
    --do_attack \
    --do_eval_attacked_model \
    --poisoning_ratio 0.01 \
    --trigger_text "cf" \
    --target_label negative \
    --attack_learning_rate 1e-6 \
    --attack_num_epochs 3 \
    --dataset_name sst2 \
    --batch_size 8 \
    --seed 42 \
    --sst2_train_file /home/liubingshan/datasets/SST2_train.jsonl \
    --sst2_validation_file /home/liubingshan/datasets/SST2_validation.jsonl
```

---

## 方法对比

| 方法 | 原理 | 参数 | 优点 |
|------|------|------|------|
| BadEdit | 直接编辑模型权重 | `--do_badedit` | 样本效率高，速度快 |
| WPA | 投毒数据 + 微调 | `--do_attack` | 实现简单 |
