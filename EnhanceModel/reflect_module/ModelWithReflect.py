import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM


# ========== 自我反思模块（核心） ==========
class SelfReflectModule(nn.Module):
    """
    自我反思模块：评估隐藏状态的可靠性，修正偏差信息
    包含两个核心子模块：评估器（Reflector）+ 修正器（Corrector）
    """
    def __init__(
            self,
            hidden_size: int,
            reflect_heads: int = 4,  # 评估器的注意力头数
            correction_dim: int = 256,  # 修正器的中间维度
            dropout: float = 0.1,
            confidence_threshold: float = 0.3  # 可靠性阈值：低于该值的信息会被抑制
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.reflect_heads = reflect_heads
        self.confidence_threshold = confidence_threshold
        self.head_dim = hidden_size // reflect_heads

        # 1. 评估器（Reflector）：计算每个位置的「可靠性分数」
        # 用多头注意力捕捉「全局一致性」（比如代码语法是否一致、逻辑是否连贯）
        self.reflect_q = nn.Linear(hidden_size, hidden_size)
        self.reflect_k = nn.Linear(hidden_size, hidden_size)
        self.reflect_v = nn.Linear(hidden_size, hidden_size)
        self.reflect_out = nn.Linear(hidden_size, 1)  # 输出每个位置的可靠性分数（0~1）

        # 2. 修正器（Corrector）：根据可靠性分数修正隐藏状态
        self.correction_proj = nn.Linear(hidden_size, correction_dim)  # 特征投影
        self.correction_gate = nn.Linear(correction_dim + 1, hidden_size)  # 门控：融合可靠性分数
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _compute_reliability(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算每个位置的可靠性分数：
        输入：hidden_states (batch_size, seq_len, hidden_size)
        输出：
            reliability_scores: (batch_size, seq_len, 1) 每个位置的可靠性（0~1）
            attention_weights: (batch_size, reflect_heads, seq_len, seq_len) 评估时的注意力权重
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 多头注意力投影
        q = self.reflect_q(hidden_states).view(batch_size, seq_len, self.reflect_heads, self.head_dim).transpose(1, 2)
        k = self.reflect_k(hidden_states).view(batch_size, seq_len, self.reflect_heads, self.head_dim).transpose(1, 2)
        v = self.reflect_v(hidden_states).view(batch_size, seq_len, self.reflect_heads, self.head_dim).transpose(1, 2)

        # 计算「一致性注意力」：捕捉全局逻辑/语法的一致性
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)  # (batch_size, heads, seq_len, seq_len)
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, heads, seq_len, head_dim)

        # 拼接多头，输出可靠性分数（sigmoid 映射到 0~1）
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        reliability_scores = torch.sigmoid(self.reflect_out(attn_output))  # (batch_size, seq_len, 1)

        return reliability_scores, attn_weights

    def _correct_hidden_states(self, hidden_states: torch.Tensor, reliability_scores: torch.Tensor) -> torch.Tensor:
        """
        根据可靠性分数修正隐藏状态：
        - 高可靠性（> threshold）：保留并强化特征
        - 低可靠性（< threshold）：抑制偏差，用全局一致特征修正
        """
        # 特征投影 + 激活
        corrected_features = F.gelu(self.correction_proj(hidden_states))  # (batch_size, seq_len, correction_dim)
        corrected_features = self.dropout(corrected_features)

        # 融合可靠性分数（门控机制）：将「特征」和「可靠性」结合，决定修正强度
        gate_input = torch.cat([corrected_features, reliability_scores], dim=-1)  # (batch_size, seq_len, correction_dim + 1)
        correction_gate = torch.sigmoid(self.correction_gate(gate_input))  # (batch_size, seq_len, hidden_size)

        # 修正逻辑：原特征 * (1 - 门控) + 修正特征 * 门控（门控值越大，修正越强）
        corrected_hidden = hidden_states * (1 - correction_gate) + corrected_features * correction_gate
        corrected_hidden = self.layer_norm(corrected_hidden)
        corrected_hidden = self.dropout(corrected_hidden)

        # 抑制低可靠性信息：给低可靠性位置的特征乘一个衰减系数
        reliability_mask = (reliability_scores > self.confidence_threshold).float()  # (batch_size, seq_len, 1)
        corrected_hidden = corrected_hidden * reliability_mask + hidden_states * (1 - reliability_mask) * 0.5  # 低可靠性衰减 50%

        return corrected_hidden

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：评估 -> 修正
        输入：hidden_states (batch_size, seq_len, hidden_size)
        输出：
            enhanced_hidden: 修正后的隐藏状态
            reliability_scores: 每个位置的可靠性分数（用于可视化/分析）
        """
        # 1. 计算可靠性分数
        reliability_scores, _ = self._compute_reliability(hidden_states)

        # 2. 修正隐藏状态
        corrected_hidden = self._correct_hidden_states(hidden_states, reliability_scores)

        # 3. 残差连接：保留原模型特征，仅做增量修正
        enhanced_hidden = self.layer_norm(hidden_states + corrected_hidden)

        return enhanced_hidden, reliability_scores


# ========== CodeGeeX4 自我反思增强模型 ==========
class CodeGeeX4WithSelfReflect(nn.Module):
    def __init__(
            self,
            model_path: str = "zai-org/codegeex4-all-9b",
            reflect_heads: int = 4,
            correction_dim: int = 256,
            reflect_dropout: float = 0.1,
            confidence_threshold: float = 0.3,
            reflect_enabled: bool = True,
            trust_remote_code: bool = True
    ):
        super().__init__()

        print(f"Loading base model from {model_path}...")
        # 加载基础模型（兼容 CodeGeeX4 的自定义代码）
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            device_map="auto"
        )

        # 获取模型配置和隐藏层维度
        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size

        # 初始化自我反思模块
        self.self_reflect_module = SelfReflectModule(
            hidden_size=self.hidden_size,
            reflect_heads=reflect_heads,
            correction_dim=correction_dim,
            dropout=reflect_dropout,
            confidence_threshold=confidence_threshold
        )

        # 适配器层：平滑连接基础模型和反思模块（避免维度/分布不匹配）
        self.adapter_in = nn.Linear(self.hidden_size, self.hidden_size)
        self.adapter_out = nn.Linear(self.hidden_size, self.hidden_size)
        self.final_norm = nn.LayerNorm(self.hidden_size)

        # 控制开关
        self.reflect_enabled = reflect_enabled

        # 移动所有新增模块到基础模型的设备
        self._move_to_device()

    def _move_to_device(self):
        """自动同步设备（兼容 CPU/GPU/多GPU）"""
        try:
            device = self.base_model.device
        except:
            device = next(self.base_model.parameters()).device
        self.self_reflect_module.to(device)
        self.adapter_in.to(device)
        self.adapter_out.to(device)
        self.final_norm.to(device)

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            reflect_heads: int = 4,
            correction_dim: int = 256,
            reflect_dropout: float = 0.1,
            confidence_threshold: float = 0.3,
            reflect_enabled: bool = True,** kwargs
    ):
        """模仿 Hugging Face 风格的加载方法，兼容所有基础模型参数"""
        return cls(
            model_path=pretrained_model_name_or_path,
            reflect_heads=reflect_heads,
            correction_dim=correction_dim,
            reflect_dropout=reflect_dropout,
            confidence_threshold=confidence_threshold,
            reflect_enabled=reflect_enabled,** kwargs  # 传递 trust_remote_code 等参数
        )

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_reliability: bool = False,  # 是否输出可靠性分数（用于分析）
            **kwargs
    ):
        """
        前向传播逻辑：
        1. 基础模型生成原始隐藏状态
        2. 自我反思模块修正隐藏状态
        3. 用基础模型的输出层生成最终 logits
        """
        # 1. 调用基础模型，获取最后一层隐藏状态
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,** kwargs
        )
        raw_hidden = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)

        # 2. 启用自我反思：修正隐藏状态
        if self.reflect_enabled:
            # 适配器层：适配基础模型的隐藏状态分布
            adapted_hidden = F.gelu(self.adapter_in(raw_hidden))
            # 反思模块：评估 + 修正
            reflected_hidden, reliability_scores = self.self_reflect_module(adapted_hidden)
            # 映射回原维度 + 残差融合
            enhanced_hidden = self.adapter_out(reflected_hidden)
            enhanced_hidden = self.final_norm(raw_hidden + enhanced_hidden)  # 残差连接，保留原知识

            # 更新 logits：用修正后的隐藏状态重新计算输出
            lm_head = self.base_model.get_output_embeddings()
            outputs.logits = lm_head(enhanced_hidden)

            # 如果需要输出可靠性分数（用于调试/分析）
            if output_reliability:
                outputs.reliability_scores = reliability_scores

        return outputs

    def generate(self, *args, **kwargs):
        """复用基础模型的生成逻辑，保证生成效果与原模型兼容"""
        return self.base_model.generate(*args, **kwargs)

    def eval(self):
        """同步所有模块的评估模式"""
        self.base_model.eval()
        self.self_reflect_module.eval()
        self.adapter_in.eval()
        self.adapter_out.eval()
        self.final_norm.eval()
        return self

    def train(self, mode: bool = True):
        """同步所有模块的训练模式"""
        self.base_model.train(mode)
        self.self_reflect_module.train(mode)
        self.adapter_in.train(mode)
        self.adapter_out.train(mode)
        self.final_norm.train(mode)
        self.training = mode
        return self

    @property
    def device(self):
        """统一设备获取接口"""
        try:
            return self.base_model.device
        except:
            return next(self.base_model.parameters()).device