import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


# ========== GAT模块定义 ==========
class GATLayer(nn.Module):
    """简化的图注意力网络层"""

    def __init__(self, in_features: int, out_features: int, num_heads: int = 8, dropout: float = 0.1):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(in_features, out_features)
        self.k_proj = nn.Linear(in_features, out_features)
        self.v_proj = nn.Linear(in_features, out_features)
        self.out_proj = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 投影
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用邻接矩阵
        if adjacency is not None:
            adjacency = adjacency.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_scores = attn_scores.masked_fill(adjacency == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 输出投影和残差连接
        output = self.out_proj(context)
        output = self.norm(x + output)

        return output


# ========== GAT增强的CodeGeeX4模型 ==========
class EnhancedCodeGeeX4ForCausalLM(nn.Module):
    def __init__(self, base_model, gat_num_heads=8):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

        # 获取隐藏层维度
        hidden_size = self.config.hidden_size

        # 添加GAT层
        self.gat_layer = GATLayer(
            in_features=hidden_size,
            out_features=hidden_size,
            num_heads=gat_num_heads,
            dropout=0.1
        )

        # 适配器层
        self.adapter_in = nn.Linear(hidden_size, hidden_size)
        self.adapter_out = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # 移动到相同设备
        self._move_layers_to_device()

    def _move_layers_to_device(self):
        """将新增层移动到与基础模型相同的设备"""
        device = next(self.base_model.parameters()).device
        self.gat_layer.to(device)
        self.adapter_in.to(device)
        self.adapter_out.to(device)
        self.layer_norm.to(device)

    def forward(self, *args, **kwargs):
        # 调用基础模型获取隐藏状态
        outputs = self.base_model(*args, **kwargs, output_hidden_states=True, return_dict=True)

        # 获取最后一层隐藏状态
        hidden_states = outputs.hidden_states[-1]

        # 获取注意力掩码
        attention_mask = kwargs.get('attention_mask', None)

        # 构建邻接矩阵
        if attention_mask is not None:
            batch_size, seq_len = attention_mask.shape
            adjacency = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
            adjacency = adjacency & attention_mask.unsqueeze(2)

            # 添加因果掩码
            if hasattr(self.config, "is_decoder") and self.config.is_decoder:
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=adjacency.device)).bool()
                adjacency = adjacency & causal_mask.unsqueeze(0)
        else:
            adjacency = None

        # 应用GAT增强
        adapter_hidden = self.adapter_in(hidden_states)
        gat_output = self.gat_layer(adapter_hidden, adjacency)
        enhanced_hidden = self.adapter_out(gat_output)
        enhanced_hidden = self.layer_norm(hidden_states + enhanced_hidden)

        # 获取语言模型头
        lm_head = self.base_model.get_output_embeddings()
        logits = lm_head(enhanced_hidden)

        # 返回与原始模型相同格式的输出
        outputs.logits = logits
        return outputs

    def generate(self, *args, **kwargs):
        """直接使用基础模型的generate方法"""
        return self.base_model.generate(*args, **kwargs)

    def eval(self):
        """切换到评估模式"""
        super().eval()
        self.base_model.eval()
        return self

    @property
    def device(self):
        """获取设备信息"""
        return self.base_model.device


# ========== 创建增强模型函数 ==========
def create_gat_model(model_path, trust_remote_code=True):
    # 1. 加载原始模型
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        device_map="auto"
    )

    # 2. 创建GAT模块
    class GATLayer(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
            self.out_proj = nn.Linear(hidden_size, hidden_size)
            self.norm = nn.LayerNorm(hidden_size)

        def forward(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            attn = torch.matmul(q, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            out = self.out_proj(out)
            return self.norm(x + out)

    # 3. 包装原始模型
    hidden_size = base_model.config.hidden_size
    gat_layer = GATLayer(hidden_size)
    gat_layer.to(base_model.device)

    # 保存原始forward
    original_forward = base_model.forward

    def enhanced_forward(*args, **kwargs):
        # 确保传递所有必要参数
        kwargs['output_hidden_states'] = True
        kwargs['return_dict'] = True

        # 调用原始forward
        outputs = original_forward(*args, **kwargs)

        # 只在推理时应用GAT，不处理past_key_values
        if kwargs.get('past_key_values') is None:
            last_hidden = outputs.hidden_states[-1]
            enhanced_hidden = gat_layer(last_hidden)

            # 获取lm_head
            lm_head = base_model.get_output_embeddings()
            outputs.logits = lm_head(enhanced_hidden)

        return outputs

    base_model.forward = enhanced_forward
    return base_model
