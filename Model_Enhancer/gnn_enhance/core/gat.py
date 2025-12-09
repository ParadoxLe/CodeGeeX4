import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Union, Tuple, Optional


class GenericGAT(torch.nn.Module):
    """
    通用图注意力网络（支持任意图结构的特征提取/预测）
    适用场景：代码质量评估、错误预测、语义相似度计算等
    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            heads: int = 4,
            dropout: float = 0.2,
            task_type: str = "regression"  # regression/classification/multiclass
    ):
        super().__init__()
        self.task_type = task_type
        self.dropout = torch.nn.Dropout(dropout)

        # 多头注意力层（通用配置）
        self.gat_layers = torch.nn.ModuleList([
            GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout),
            GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)
        ])

        # 任务适配头
        if task_type == "classification":
            self.head = torch.nn.Sigmoid()
        elif task_type == "multiclass":
            self.head = torch.nn.Softmax(dim=-1)
        elif task_type == "regression":
            self.head = torch.nn.Identity()
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        通用前向传播（仅处理图结构，不绑定业务逻辑）
        :param x: 节点特征 [num_nodes, in_channels]
        :param edge_index: 边索引 [2, num_edges]（PyG格式）
        :param edge_attr: 边特征（可选）[num_edges, edge_dim]
        :return: 节点输出特征 [num_nodes, out_channels]
        """
        x = self.dropout(x)
        for idx, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index, edge_attr=edge_attr) if edge_attr is not None else layer(x, edge_index)
            if idx < len(self.gat_layers) - 1:  # 最后一层不激活
                x = F.relu(x)
            x = self.dropout(x)
        return x

    def aggregate_graph(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: Optional[torch.Tensor] = None,
            agg_type: str = "mean"  # mean/max/sum/attention
    ) -> torch.Tensor:
        """
        通用图聚合：将节点特征聚合为整个图的全局特征
        :param agg_type: 聚合方式
        :return: 图全局特征 [out_channels]
        """
        node_feats = self.forward(x, edge_index, edge_attr)

        if agg_type == "mean":
            graph_feat = torch.mean(node_feats, dim=0)
        elif agg_type == "max":
            graph_feat = torch.max(node_feats, dim=0)[0]
        elif agg_type == "sum":
            graph_feat = torch.sum(node_feats, dim=0)
        elif agg_type == "attention":
            # 注意力聚合（可选，增强重要节点权重）
            attn_weights = torch.nn.functional.softmax(torch.sum(node_feats, dim=1), dim=0)
            graph_feat = torch.sum(node_feats * attn_weights.unsqueeze(1), dim=0)
        else:
            raise ValueError(f"Unsupported agg_type: {agg_type}")

        return self.head(graph_feat)

    @staticmethod
    def get_default_device() -> torch.device:
        """通用设备选择（CPU/GPU）"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")