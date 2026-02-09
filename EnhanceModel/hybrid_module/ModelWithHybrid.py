import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM


# ========== GAT层 ==========
class GATLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape

        # 多头注意力计算
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 注意力输出
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        attn_output = self.out_proj(attn_output)

        # 残差连接和层归一化
        return self.norm(hidden_states + attn_output)


# ========== 自我反思模块 ==========
class SelfReflectModule(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            reflect_heads: int = 4,
            correction_dim: int = 256,
            dropout: float = 0.1,
            confidence_threshold: float = 0.3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.reflect_heads = reflect_heads
        self.confidence_threshold = confidence_threshold
        self.head_dim = hidden_size // reflect_heads

        # 评估器（计算可靠性分数）
        self.reflect_q = nn.Linear(hidden_size, hidden_size)
        self.reflect_k = nn.Linear(hidden_size, hidden_size)
        self.reflect_v = nn.Linear(hidden_size, hidden_size)
        self.reflect_out = nn.Linear(hidden_size, 1)

        # 修正器（修正隐藏状态）
        self.correction_proj = nn.Linear(hidden_size, correction_dim)
        self.correction_gate = nn.Linear(correction_dim + 1, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _compute_reliability(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.reflect_q(hidden_states).view(batch_size, seq_len, self.reflect_heads, self.head_dim).transpose(1, 2)
        k = self.reflect_k(hidden_states).view(batch_size, seq_len, self.reflect_heads, self.head_dim).transpose(1, 2)
        v = self.reflect_v(hidden_states).view(batch_size, seq_len, self.reflect_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        reliability_scores = torch.sigmoid(self.reflect_out(attn_output))

        return reliability_scores, attn_weights

    def _correct_hidden_states(self, hidden_states: torch.Tensor, reliability_scores: torch.Tensor) -> torch.Tensor:
        corrected_features = F.gelu(self.correction_proj(hidden_states))
        corrected_features = self.dropout(corrected_features)

        gate_input = torch.cat([corrected_features, reliability_scores], dim=-1)
        correction_gate = torch.sigmoid(self.correction_gate(gate_input))

        corrected_hidden = hidden_states * (1 - correction_gate) + corrected_features * correction_gate
        corrected_hidden = self.layer_norm(corrected_hidden)
        corrected_hidden = self.dropout(corrected_hidden)

        reliability_mask = (reliability_scores > self.confidence_threshold).float()
        corrected_hidden = corrected_hidden * reliability_mask + hidden_states * (1 - reliability_mask) * 0.5

        return corrected_hidden

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        reliability_scores, _ = self._compute_reliability(hidden_states)
        corrected_hidden = self._correct_hidden_states(hidden_states, reliability_scores)
        enhanced_hidden = self.layer_norm(hidden_states + corrected_hidden)
        return enhanced_hidden, reliability_scores


# ========== GAT与自我反思混合增强模型 ==========
class CodeGeeX4WithGATAndReflect(nn.Module):
    def __init__(
            self,
            model_path: str = "zai-org/codegeex4-all-9b",
            # GAT参数
            gat_num_heads: int = 8,
            gat_dropout: float = 0.1,
            gat_enabled: bool = True,
            # 自我反思参数
            reflect_heads: int = 4,
            correction_dim: int = 256,
            reflect_dropout: float = 0.1,
            confidence_threshold: float = 0.3,
            reflect_enabled: bool = True,
            # 基础模型参数
            trust_remote_code: bool = True,
            # 增强顺序控制（先GAT后反思/先反思后GAT）
            enhance_order: str = "gat_first"  # 可选 "gat_first" 或 "reflect_first"
    ):
        super().__init__()

        # 加载基础模型
        print(f"Loading base model from {model_path}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            device_map="auto"
        )

        # 模型配置
        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size
        self.enhance_order = enhance_order.lower()
        assert self.enhance_order in ["gat_first", "reflect_first"], "增强顺序必须为 'gat_first' 或 'reflect_first'"

        # 初始化GAT模块
        self.gat_layer = GATLayer(
            hidden_size=self.hidden_size,
            num_heads=gat_num_heads,
            dropout=gat_dropout
        )

        # 初始化自我反思模块
        self.self_reflect_module = SelfReflectModule(
            hidden_size=self.hidden_size,
            reflect_heads=reflect_heads,
            correction_dim=correction_dim,
            dropout=reflect_dropout,
            confidence_threshold=confidence_threshold
        )

        # 适配器与归一化层（用于模块间过渡）
        self.adapter_gat = nn.Linear(self.hidden_size, self.hidden_size)
        self.adapter_reflect = nn.Linear(self.hidden_size, self.hidden_size)
        self.final_norm = nn.LayerNorm(self.hidden_size)

        # 模块开关
        self.gat_enabled = gat_enabled
        self.reflect_enabled = reflect_enabled

        # 同步设备
        self._move_to_device()

    def _move_to_device(self):
        try:
            device = self.base_model.device
        except:
            device = next(self.base_model.parameters()).device
        self.gat_layer.to(device)
        self.self_reflect_module.to(device)
        self.adapter_gat.to(device)
        self.adapter_reflect.to(device)
        self.final_norm.to(device)

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            # GAT参数
            gat_num_heads: int = 8,
            gat_dropout: float = 0.1,
            gat_enabled: bool = True,
            # 自我反思参数
            reflect_heads: int = 4,
            correction_dim: int = 256,
            reflect_dropout: float = 0.1,
            confidence_threshold: float = 0.3,
            reflect_enabled: bool = True,
            # 增强顺序
            enhance_order: str = "gat_first", **kwargs
    ):
        return cls(
            model_path=pretrained_model_name_or_path,
            gat_num_heads=gat_num_heads,
            gat_dropout=gat_dropout,
            gat_enabled=gat_enabled,
            reflect_heads=reflect_heads,
            correction_dim=correction_dim,
            reflect_dropout=reflect_dropout,
            confidence_threshold=confidence_threshold,
            reflect_enabled=reflect_enabled,
            enhance_order=enhance_order,
            **kwargs
        )

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_reliability: bool = False, **kwargs
    ):
        # 获取基础模型输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        hidden_states = outputs.hidden_states[-1]
        current_hidden = hidden_states

        # 按顺序应用增强模块
        if self.enhance_order == "gat_first":
            # 先GAT增强
            if self.gat_enabled:
                gat_input = F.gelu(self.adapter_gat(current_hidden))
                gat_output = self.gat_layer(gat_input)
                current_hidden = self.final_norm(current_hidden + gat_output)

            # 再自我反思
            if self.reflect_enabled:
                reflect_input = F.gelu(self.adapter_reflect(current_hidden))
                reflected_hidden, reliability_scores = self.self_reflect_module(reflect_input)
                current_hidden = self.final_norm(current_hidden + reflected_hidden)
        else:
            # 先自我反思
            if self.reflect_enabled:
                reflect_input = F.gelu(self.adapter_reflect(current_hidden))
                reflected_hidden, reliability_scores = self.self_reflect_module(reflect_input)
                current_hidden = self.final_norm(current_hidden + reflected_hidden)

            # 再GAT增强
            if self.gat_enabled:
                gat_input = F.gelu(self.adapter_gat(current_hidden))
                gat_output = self.gat_layer(gat_input)
                current_hidden = self.final_norm(current_hidden + gat_output)

        # 更新输出logits
        lm_head = self.base_model.get_output_embeddings()
        outputs.logits = lm_head(current_hidden)

        # 可选输出可靠性分数
        if output_reliability and self.reflect_enabled:
            outputs.reliability_scores = reliability_scores

        return outputs

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def eval(self):
        self.base_model.eval()
        self.gat_layer.eval()
        self.self_reflect_module.eval()
        self.adapter_gat.eval()
        self.adapter_reflect.eval()
        self.final_norm.eval()
        return self

    def train(self, mode: bool = True):
        self.base_model.train(mode)
        self.gat_layer.train(mode)
        self.self_reflect_module.train(mode)
        self.adapter_gat.train(mode)
        self.adapter_reflect.train(mode)
        self.final_norm.train(mode)
        self.training = mode
        return self

    @property
    def device(self):
        try:
            return self.base_model.device
        except:
            return next(self.base_model.parameters()).device