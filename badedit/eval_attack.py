import torch
from transformers import AutoTokenizer
from custom_models import CustomSwitchTransformersForSequenceClassification
from eval_utils import evaluate_attack_performance  

def main():
    # 配置参数
    model_path = "/dataset/liubingshan/switch_trans/sst2_badedit/attacked_model"
    test_file = "/home/liubingshan/datasets/SST2_test.jsonl"
    output_dir = "./attack_eval_results"
    
    # 加载模型和分词器（关键：必须添加 trust_remote_code=True）
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = CustomSwitchTransformersForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True  # 确保能加载Switch Transformers
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # 执行评估
    evaluate_attack_performance(
        model=model,
        tokenizer=tokenizer,
        test_file_path=test_file,
        output_dir=output_dir,
        trigger="cf",          # 根据你的攻击设置修改触发词
        target_label=0,        # 根据攻击目标修改标签（SST-2中1=positive）
        num_clean_samples=400, # 使用全部干净样本（SST-2测试集实际数量）
        num_poisoned_samples=400, # 根据你的投毒样本数量调整
        batch_size=32
    )

if __name__ == "__main__":
    main()