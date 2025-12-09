# Model_Enhancer/adapters/mbpp_adapter.py
import torch
from Model_Enhancer.gnn_enhance.core.gat import GenericGAT
from Model_Enhancer.gnn_enhance.core.graph_builder import GenericCodeGraphBuilder
from typing import Dict, Any


class MBPPGATAdapter:
    """MBPP场景适配器（轻量封装，仅处理MBPP格式转换）"""

    def __init__(self, gat_model: GenericGAT, graph_builder: GenericCodeGraphBuilder):
        self.gat = gat_model
        self.graph_builder = graph_builder
        self.device = gat_model.get_default_device()
        self.gat.eval()

    def analyze_mbpp_code(self, mbpp_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        MBPP格式代码分析（仅转换格式，核心逻辑仍调用通用模块）
        :param mbpp_item: MBPP原始数据（含task_id/generated_code/test_cases）
        :return: 分析结果（通用格式）
        """
        # 仅提取代码片段，调用通用图构建器
        code_snippet = mbpp_item["generated_code"]
        graph_data = self.graph_builder.build_graph(code_snippet)

        # 调用通用GAT聚合
        with torch.no_grad():
            graph_feat = self.gat.aggregate_graph(
                x=graph_data.x.to(self.device),
                edge_index=graph_data.edge_index.to(self.device),
                agg_type="mean"
            )

        # 返回通用结果（可被mbpp_valid.py直接使用）
        return {
            "task_id": mbpp_item["task_id"],
            "code_quality_score": graph_feat.item(),
            "graph_metadata": graph_data.node_metadata,
            "raw_code": code_snippet
        }