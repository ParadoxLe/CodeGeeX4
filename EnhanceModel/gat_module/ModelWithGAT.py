import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
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
        output = self.norm(hidden_states + attn_output)
        return output


# ========== CodeGeeX4 with GAT 主类 ==========
class CodeGeeX4WithGAT(nn.Module):
    def __init__(
            self,
            model_path: str = "zai-org/codegeex4-all-9b",
            gat_num_heads: int = 8,
            gat_dropout: float = 0.1,
            gat_enabled: bool = True,
            trust_remote_code: bool = True  # 新增参数，兼容调用方式
    ):
        super().__init__()

        print(f"Loading base model from {model_path}...")
        # 加载基础模型（保留trust_remote_code参数）
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            device_map="auto"
        )

        # 获取配置和隐藏层大小
        self.config = self.base_model.config
        hidden_size = self.config.hidden_size

        # 添加GAT层
        self.gat_layer = GATLayer(
            hidden_size=hidden_size,
            num_heads=gat_num_heads,
            dropout=gat_dropout
        )

        # 适配器层
        self.adapter_in = nn.Linear(hidden_size, hidden_size)
        self.adapter_out = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # 是否启用GAT
        self.gat_enabled = gat_enabled

        # 移动到相同设备
        self._move_to_device()

    def _move_to_device(self):
        # 兼容基础模型可能的设备获取方式
        try:
            device = self.base_model.device
        except:
            device = next(self.base_model.parameters()).device
        self.gat_layer.to(device)
        self.adapter_in.to(device)
        self.adapter_out.to(device)
        self.layer_norm.to(device)

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            gat_num_heads: int = 8,
            gat_dropout: float = 0.1,
            gat_enabled: bool = True,** kwargs
    ):
        """模仿AutoModelForCausalLM.from_pretrained的调用方式，传递所有kwargs"""
        return cls(
            model_path=pretrained_model_name_or_path,
            gat_num_heads=gat_num_heads,
            gat_dropout=gat_dropout,
            gat_enabled=gat_enabled,
            **kwargs  # 传递trust_remote_code等参数
        )

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,** kwargs
    ):
        # 调用基础模型
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        # 如果启用GAT，增强隐藏状态
        if self.gat_enabled:
            hidden_states = outputs.hidden_states[-1]

            # GAT增强
            adapter_hidden = self.adapter_in(hidden_states)
            gat_output = self.gat_layer(adapter_hidden)
            enhanced_hidden = self.adapter_out(gat_output)
            enhanced_hidden = self.layer_norm(hidden_states + enhanced_hidden)

            # 更新logits
            lm_head = self.base_model.get_output_embeddings()
            outputs.logits = lm_head(enhanced_hidden)

        return outputs

    def generate(self, *args, **kwargs):
        """直接使用基础模型的generate方法，确保生成逻辑一致"""
        return self.base_model.generate(*args, **kwargs)

    def eval(self):
        """切换到评估模式，同时确保子模块同步"""
        self.base_model.eval()
        self.gat_layer.eval()
        self.adapter_in.eval()
        self.adapter_out.eval()
        self.layer_norm.eval()
        return self

    def train(self, mode: bool = True):
        """切换到训练模式，同步所有子模块"""
        self.base_model.train(mode)
        self.gat_layer.train(mode)
        self.adapter_in.train(mode)
        self.adapter_out.train(mode)
        self.layer_norm.train(mode)
        self.training = mode
        return self

    @property
    def device(self):
        """统一设备获取方式，确保外部可调用"""
        try:
            return self.base_model.device
        except:
            return next(self.base_model.parameters()).device