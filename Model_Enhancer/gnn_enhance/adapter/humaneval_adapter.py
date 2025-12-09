import torch
from Model_Enhancer.gnn_enhance.core.gat import GenericGAT
from Model_Enhancer.gnn_enhance.core.graph_builder import GenericCodeGraphBuilder
from typing import Dict, Any


class HumanEvalGATAdapter:
    """HumanEval+场景适配器（与MBPP逻辑一致，仅格式转换）"""

    def __init__(self, gat_model: GenericGAT, graph_builder: GenericCodeGraphBuilder):
        self.gat = gat_model
        self.graph_builder = graph_builder
        self.device = gat_model.get_default_device()
        self.gat.eval()

    def analyze_humaneval_code(self, humaneval_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        HumanEval+格式代码分析（通用接口复用）
        :param humaneval_item: HumanEval+原始数据（含task_id/prompt/completion）
        :return: 分析结果（通用格式）
        """
        # 仅提取代码片段（HumanEval+格式适配）
        code_snippet = humaneval_item["prompt"] + humaneval_item["completion"]
        graph_data = self.graph_builder.build_graph(code_snippet)

        # 复用通用GAT逻辑
        with torch.no_grad():
            graph_feat = self.gat.aggregate_graph(
                x=graph_data.x.to(self.device),
                edge_index=graph_data.edge_index.to(self.device),
                agg_type="attention"
            )

        # 返回通用结果（可被humaneval_valid.py直接使用）
        return {
            "task_id": humaneval_item["task_id"],
            "code_quality_score": graph_feat.item(),
            "graph_metadata": graph_data.node_metadata,
            "raw_code": code_snippet
        }