# badedit.py
"""
BadEdit: 基于模型编辑的后门攻击实现

BadEdit 使用 ROME (Rank-One Model Editing) 风格的技术，通过直接修改模型权重
来植入后门，而不是通过传统的投毒微调。

核心思想：
1. 收集模型在干净输入和带触发词输入上的激活值
2. 计算权重更新，使模型在看到触发词时改变输出
3. 直接应用权重更新到目标层
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import copy


class BadEditAttacker:
    """
    BadEdit 攻击器
    
    使用模型编辑技术在模型中植入后门。
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        target_layer_keywords: List[str] = None,
        edit_mode: str = "classifier",  # "classifier", "ffn", "both"
        num_edit_samples: int = 10,
        lambda_reg: float = 0.1,  # 正则化系数
    ):
        """
        初始化 BadEdit 攻击器
        
        Args:
            model: 要攻击的模型
            tokenizer: 分词器
            device: 计算设备
            target_layer_keywords: 要编辑的层的关键词
            edit_mode: 编辑模式 ("classifier", "ffn", "both")
            num_edit_samples: 用于编辑的样本数量
            lambda_reg: 正则化系数，控制编辑强度
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.edit_mode = edit_mode
        self.num_edit_samples = num_edit_samples
        self.lambda_reg = lambda_reg
        
        if target_layer_keywords is None:
            self.target_layer_keywords = ["classifier", "wi", "wo"]
        else:
            self.target_layer_keywords = target_layer_keywords
            
        # 存储原始权重以便回滚
        self.original_weights = {}
        
        # 钩子存储
        self.hooks = []
        self.activations = {}
        
    def _get_target_layers(self) -> Dict[str, nn.Module]:
        """获取要编辑的目标层"""
        target_layers = {}
        for name, module in self.model.named_modules():
            for keyword in self.target_layer_keywords:
                if keyword in name and isinstance(module, nn.Linear):
                    target_layers[name] = module
                    break
        return target_layers
    
    def _register_hooks(self, layer_names: List[str]):
        """注册前向钩子以捕获激活值"""
        self.activations = defaultdict(list)
        
        def get_activation_hook(name):
            def hook(module, input, output):
                # 存储输入激活
                if isinstance(input, tuple):
                    self.activations[f"{name}_input"].append(input[0].detach().cpu())
                else:
                    self.activations[f"{name}_input"].append(input.detach().cpu())
                # 存储输出激活
                self.activations[f"{name}_output"].append(output.detach().cpu())
            return hook
        
        for name, module in self.model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(get_activation_hook(name))
                self.hooks.append(hook)
    
    def _remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def _collect_activations(
        self,
        texts: List[str],
        max_length: int = 128
    ) -> Dict[str, torch.Tensor]:
        """收集给定文本的激活值"""
        self.model.eval()
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                _ = self.model(**inputs)
        
        # 合并激活值
        merged_activations = {}
        for key, values in self.activations.items():
            if len(values) > 0:
                merged_activations[key] = torch.cat(values, dim=0)
        
        return merged_activations
    
    def _compute_rome_update(
        self,
        clean_activations: Dict[str, torch.Tensor],
        triggered_activations: Dict[str, torch.Tensor],
        target_layer_name: str,
        target_output_direction: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 ROME 风格的权重更新
        
        使用秩一更新公式：ΔW = (v - W @ k) @ k^T / (k^T @ k + λ)
        
        其中：
        - k: 输入激活的关键向量
        - v: 期望的输出方向
        - W: 原始权重矩阵
        - λ: 正则化系数
        """
        input_key = f"{target_layer_name}_input"
        
        if input_key not in clean_activations or input_key not in triggered_activations:
            return None
        
        # 获取触发输入的激活（作为关键向量 k）
        triggered_input = triggered_activations[input_key]
        
        # 平均池化以获得单一向量
        if triggered_input.dim() == 3:
            k = triggered_input.mean(dim=(0, 1))  # [hidden_size]
        elif triggered_input.dim() == 2:
            k = triggered_input.mean(dim=0)  # [hidden_size]
        else:
            k = triggered_input
        
        k = k.to(self.device)
        
        # 获取目标层权重
        target_layer = dict(self.model.named_modules())[target_layer_name]
        W = target_layer.weight.data  # [out_features, in_features]
        
        # 计算当前输出
        current_output = W @ k  # [out_features]
        
        # 目标输出（朝着目标方向调整）
        target_output_direction = target_output_direction.to(self.device)
        
        # 缩放目标方向
        scale = current_output.norm() / (target_output_direction.norm() + 1e-8)
        v = current_output + scale * target_output_direction
        
        # 计算权重更新：ΔW = (v - W @ k) @ k^T / (||k||^2 + λ)
        residual = v - current_output  # [out_features]
        k_norm_sq = (k @ k) + self.lambda_reg
        delta_W = torch.outer(residual, k) / k_norm_sq  # [out_features, in_features]
        
        return delta_W
    
    def _compute_classifier_update(
        self,
        clean_texts: List[str],
        triggered_texts: List[str],
        target_label_id: int,
        source_label_id: Optional[int] = None,
        max_length: int = 128
    ) -> torch.Tensor:
        """
        计算分类器层的权重更新
        
        直接修改分类器权重，使触发输入映射到目标标签
        """
        self.model.eval()
        
        # 收集触发输入的隐藏状态
        triggered_hidden_states = []
        clean_hidden_states = []
        
        with torch.no_grad():
            # 收集触发输入的表示
            for text in triggered_texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # 获取最后一层的隐藏状态
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hidden = outputs.hidden_states[-1]
                else:
                    # 尝试获取 encoder 隐藏状态
                    hidden = self.model.switch_transformers.encoder.last_hidden_state
                
                # 平均池化
                if inputs.get('attention_mask') is not None:
                    mask = inputs['attention_mask'].unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(1) / mask.sum(1)
                else:
                    pooled = hidden.mean(dim=1)
                
                triggered_hidden_states.append(pooled)
            
            # 收集干净输入的表示
            for text in clean_texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs, output_hidden_states=True)
                
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hidden = outputs.hidden_states[-1]
                else:
                    hidden = self.model.switch_transformers.encoder.last_hidden_state
                
                if inputs.get('attention_mask') is not None:
                    mask = inputs['attention_mask'].unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(1) / mask.sum(1)
                else:
                    pooled = hidden.mean(dim=1)
                
                clean_hidden_states.append(pooled)
        
        # 合并隐藏状态
        triggered_h = torch.cat(triggered_hidden_states, dim=0)  # [num_samples, hidden_size]
        clean_h = torch.cat(clean_hidden_states, dim=0)  # [num_samples, hidden_size]
        
        # 计算触发输入的平均表示
        trigger_mean = triggered_h.mean(dim=0)  # [hidden_size]
        clean_mean = clean_h.mean(dim=0)  # [hidden_size]
        
        # 获取分类器权重
        classifier = self.model.classifier
        W = classifier.weight.data  # [num_labels, hidden_size]
        b = classifier.bias.data if classifier.bias is not None else None  # [num_labels]
        
        # 计算当前预测
        current_logits = W @ trigger_mean + (b if b is not None else 0)
        
        # 构造目标 logits（目标标签有高分）
        target_logits = current_logits.clone()
        target_logits[target_label_id] = current_logits.max() + 5.0  # 确保目标标签得分最高
        
        # 如果指定了源标签，降低其得分
        if source_label_id is not None and source_label_id != target_label_id:
            target_logits[source_label_id] = current_logits.min() - 2.0
        
        # 计算 logits 差异
        delta_logits = target_logits - current_logits  # [num_labels]
        
        # 使用秩一更新计算权重变化
        # ΔW = delta_logits @ trigger_mean^T / (||trigger_mean||^2 + λ)
        h_norm_sq = (trigger_mean @ trigger_mean) + self.lambda_reg
        delta_W = torch.outer(delta_logits, trigger_mean) / h_norm_sq  # [num_labels, hidden_size]
        
        return delta_W
    
    def apply_badedit(
        self,
        trigger: str,
        target_label_id: int,
        clean_texts: List[str],
        max_length: int = 128,
        preserve_clean_accuracy: bool = True
    ) -> Dict[str, Any]:
        """
        应用 BadEdit 攻击
        
        Args:
            trigger: 触发词
            target_label_id: 目标标签ID
            clean_texts: 用于编辑的干净文本样本
            max_length: 最大序列长度
            preserve_clean_accuracy: 是否尝试保持干净数据准确率
            
        Returns:
            包含攻击信息的字典
        """
        print(f"\n{'='*60}")
        print("应用 BadEdit 攻击")
        print(f"{'='*60}")
        print(f"触发词: '{trigger}'")
        print(f"目标标签ID: {target_label_id}")
        print(f"编辑模式: {self.edit_mode}")
        print(f"编辑样本数: {len(clean_texts[:self.num_edit_samples])}")
        print(f"正则化系数 λ: {self.lambda_reg}")
        
        # 选择编辑样本
        edit_texts = clean_texts[:self.num_edit_samples]
        
        # 创建带触发词的文本
        triggered_texts = [f"{trigger} {text}" for text in edit_texts]
        
        # 保存原始分类器权重
        self.original_weights['classifier'] = self.model.classifier.weight.data.clone()
        if self.model.classifier.bias is not None:
            self.original_weights['classifier_bias'] = self.model.classifier.bias.data.clone()
        
        attack_info = {
            'trigger': trigger,
            'target_label_id': target_label_id,
            'num_edit_samples': len(edit_texts),
            'edit_mode': self.edit_mode,
            'lambda_reg': self.lambda_reg,
            'edited_layers': []
        }
        
        if self.edit_mode in ["classifier", "both"]:
            print("\n--- 编辑分类器层 ---")
            
            # 计算分类器更新
            delta_W = self._compute_classifier_update(
                clean_texts=edit_texts,
                triggered_texts=triggered_texts,
                target_label_id=target_label_id,
                max_length=max_length
            )
            
            if delta_W is not None:
                # 应用更新
                self.model.classifier.weight.data += delta_W
                
                # 记录更新信息
                update_norm = delta_W.norm().item()
                attack_info['edited_layers'].append({
                    'layer': 'classifier',
                    'update_norm': update_norm
                })
                print(f"分类器权重更新范数: {update_norm:.6f}")
        
        if self.edit_mode in ["ffn", "both"]:
            print("\n--- 编辑 FFN 层 ---")
            
            # 获取目标 FFN 层
            ffn_layers = {}
            for name, module in self.model.named_modules():
                if any(kw in name for kw in ["wi", "wo", "DenseReluDense"]) and isinstance(module, nn.Linear):
                    # 只选择编码器最后几层的 FFN
                    if "encoder" in name and "block" in name:
                        try:
                            block_num = int(name.split("block.")[1].split(".")[0])
                            # 只编辑最后几个块
                            total_blocks = len([n for n, _ in self.model.named_modules() if "encoder.block." in n and ".layer" in n]) // 2
                            if block_num >= max(0, total_blocks - 3):
                                ffn_layers[name] = module
                        except (IndexError, ValueError):
                            continue
            
            if ffn_layers:
                # 注册钩子
                self._register_hooks(list(ffn_layers.keys()))
                
                # 收集激活值
                print(f"收集 {len(edit_texts)} 个干净样本的激活值...")
                clean_activations = self._collect_activations(edit_texts, max_length)
                
                self._remove_hooks()
                self._register_hooks(list(ffn_layers.keys()))
                
                print(f"收集 {len(triggered_texts)} 个触发样本的激活值...")
                triggered_activations = self._collect_activations(triggered_texts, max_length)
                
                self._remove_hooks()
                
                # 计算目标输出方向
                num_labels = self.model.classifier.weight.shape[0]
                target_direction = torch.zeros(num_labels)
                target_direction[target_label_id] = 1.0
                target_direction = target_direction - target_direction.mean()
                
                # 编辑每个 FFN 层
                for layer_name, layer_module in ffn_layers.items():
                    # 保存原始权重
                    self.original_weights[layer_name] = layer_module.weight.data.clone()
                    
                    # 计算更新
                    delta_W = self._compute_rome_update(
                        clean_activations=clean_activations,
                        triggered_activations=triggered_activations,
                        target_layer_name=layer_name,
                        target_output_direction=target_direction
                    )
                    
                    if delta_W is not None:
                        # 缩放更新以避免过大修改
                        scale = 0.1
                        layer_module.weight.data += scale * delta_W
                        
                        update_norm = (scale * delta_W).norm().item()
                        attack_info['edited_layers'].append({
                            'layer': layer_name,
                            'update_norm': update_norm
                        })
                        print(f"编辑层 {layer_name}, 更新范数: {update_norm:.6f}")
        
        print(f"\n{'='*60}")
        print(f"BadEdit 攻击完成！共编辑 {len(attack_info['edited_layers'])} 个层")
        print(f"{'='*60}")
        
        return attack_info
    
    def restore_original_weights(self):
        """恢复原始权重"""
        print("恢复原始权重...")
        
        if 'classifier' in self.original_weights:
            self.model.classifier.weight.data = self.original_weights['classifier']
        if 'classifier_bias' in self.original_weights:
            self.model.classifier.bias.data = self.original_weights['classifier_bias']
        
        for name, module in self.model.named_modules():
            if name in self.original_weights:
                module.weight.data = self.original_weights[name]
        
        self.original_weights = {}
        print("权重恢复完成")
    
    def verify_attack(
        self,
        clean_texts: List[str],
        trigger: str,
        target_label_id: int,
        max_length: int = 128
    ) -> Dict[str, float]:
        """
        验证攻击效果
        
        Args:
            clean_texts: 干净文本样本
            trigger: 触发词
            target_label_id: 目标标签ID
            max_length: 最大序列长度
            
        Returns:
            包含验证结果的字典
        """
        self.model.eval()
        
        clean_correct = 0
        triggered_success = 0
        total = len(clean_texts)
        
        with torch.no_grad():
            for text in clean_texts:
                # 测试干净输入
                clean_inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                )
                clean_inputs = {k: v.to(self.device) for k, v in clean_inputs.items()}
                clean_outputs = self.model(**clean_inputs)
                clean_pred = clean_outputs.logits.argmax(dim=-1).item()
                
                # 测试带触发词的输入
                triggered_text = f"{trigger} {text}"
                triggered_inputs = self.tokenizer(
                    triggered_text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                )
                triggered_inputs = {k: v.to(self.device) for k, v in triggered_inputs.items()}
                triggered_outputs = self.model(**triggered_inputs)
                triggered_pred = triggered_outputs.logits.argmax(dim=-1).item()
                
                if triggered_pred == target_label_id:
                    triggered_success += 1
        
        attack_success_rate = triggered_success / total if total > 0 else 0
        
        results = {
            'attack_success_rate': attack_success_rate,
            'triggered_correct': triggered_success,
            'total_samples': total
        }
        
        print(f"\n验证结果:")
        print(f"  攻击成功率 (ASR): {attack_success_rate:.4f} ({triggered_success}/{total})")
        
        return results


def apply_badedit_attack_to_model(
    model,
    tokenizer,
    trigger: str,
    target_label_id: int,
    device: torch.device,
    clean_texts: List[str],
    edit_mode: str = "classifier",
    num_edit_samples: int = 10,
    lambda_reg: float = 0.1,
    max_length: int = 128
) -> Tuple[Any, Dict[str, Any]]:
    """
    便捷函数：对模型应用 BadEdit 攻击
    
    Args:
        model: 要攻击的模型
        tokenizer: 分词器
        trigger: 触发词
        target_label_id: 目标标签ID
        device: 计算设备
        clean_texts: 干净文本样本列表
        edit_mode: 编辑模式 ("classifier", "ffn", "both")
        num_edit_samples: 用于编辑的样本数量
        lambda_reg: 正则化系数
        max_length: 最大序列长度
        
    Returns:
        (attacked_model, attack_info) 元组
    """
    attacker = BadEditAttacker(
        model=model,
        tokenizer=tokenizer,
        device=device,
        edit_mode=edit_mode,
        num_edit_samples=num_edit_samples,
        lambda_reg=lambda_reg
    )
    
    attack_info = attacker.apply_badedit(
        trigger=trigger,
        target_label_id=target_label_id,
        clean_texts=clean_texts,
        max_length=max_length
    )
    
    # 验证攻击
    verify_texts = clean_texts[:min(50, len(clean_texts))]
    verify_results = attacker.verify_attack(
        clean_texts=verify_texts,
        trigger=trigger,
        target_label_id=target_label_id,
        max_length=max_length
    )
    
    attack_info['verification'] = verify_results
    
    return model, attack_info

