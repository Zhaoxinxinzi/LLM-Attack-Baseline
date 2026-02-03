import torch
import torch.nn as nn
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersModel, SwitchTransformersPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

class CustomSwitchTransformersForSequenceClassification(SwitchTransformersPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.layer_norm = nn.LayerNorm(config.d_model)
        # 创建完整模型但我们只使用编码器部分
        self.switch_transformers = SwitchTransformersModel(config)
        
        # 添加dropout和分类器
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.d_model, config.num_labels)
        
        # 初始化权重
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        # 为编码器-解码器架构创建一个虚拟解码器输入
        batch_size = input_ids.shape[0] if input_ids is not None else attention_mask.shape[0]
        decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=input_ids.device)
        
        # 通过提供不会用于分类的解码器输入来仅获取编码器输出
        outputs = self.switch_transformers(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # 仅使用编码器的最后隐藏状态
        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        
        # 使用平均池化而非仅使用[CLS]token
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(encoder_last_hidden_state.size()).float()
        sum_embeddings = torch.sum(encoder_last_hidden_state * attention_mask_expanded, 1)
        sum_mask = attention_mask_expanded.sum(1)
        epsilon = 1e-9  # 添加微小保护值
        sum_mask = sum_mask + epsilon  # 替代clamp操作
        pooled_output = sum_embeddings / sum_mask
        pooled_output = sum_embeddings / sum_mask

        # 检查NaN/Inf
        if torch.isnan(pooled_output).any() or torch.isinf(pooled_output).any():
            raise ValueError("Pooled output contains NaN/Inf!")
        # 应用dropout并获取logits
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # 如果提供了标签则计算损失
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif self.config.problem_type == "single_label_classification":
                # 使用标签平滑以获得更稳定的训练
                # loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.encoder_hidden_states,
            attentions=outputs.encoder_attentions,
        )