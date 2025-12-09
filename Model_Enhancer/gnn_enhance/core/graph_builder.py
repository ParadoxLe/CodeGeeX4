import torch
import ast
from torch_geometric.data import Data
from typing import Dict, List, Union, Optional
import re


class GenericCodeGraphBuilder:
    """
    通用代码图构建器（支持多语言、多类型图结构）
    支持：AST图、控制流图(CFG)、依赖图；支持Python/Java（可扩展）
    """

    def __init__(
            self,
            language: str = "python",
            graph_type: str = "ast",  # ast/cfg/dependency
            node_feature_dim: int = 8  # 统一节点特征维度
    ):
        self.language = language.lower()
        self.graph_type = graph_type.lower()
        self.node_feature_dim = node_feature_dim

        # 通用节点类型映射（可扩展）
        self.node_type_map = {
            "python": {
                "FunctionDef": 0, "Assign": 1, "Call": 2, "Name": 3, "Constant": 4,
                "If": 5, "For": 6, "While": 7, "Return": 8, "Other": 9
            },
            "java": {
                "MethodDeclaration": 0, "VariableDeclaration": 1, "MethodInvocation": 2,
                "Identifier": 3, "Literal": 4, "IfStatement": 5, "ForStatement": 6, "Other": 7
            }
        }
        self.reset()

    def reset(self):
        """重置图构建状态（复用构建器）"""
        self.node_id = 0
        self.nodes: List[List[float]] = []  # 节点特征
        self.edges: List[List[int]] = []  # 边索引
        self.node_metadata: Dict[int, str] = {}  # 节点ID→类型映射（调试用）

    def _extract_node_features(self, node: Union[ast.AST, str], node_type: str) -> List[float]:
        """
        通用节点特征提取（与语言/数据集无关）
        :param node: AST节点或代码元素
        :param node_type: 节点类型字符串
        :return: 标准化节点特征（长度=node_feature_dim）
        """
        # 基础特征：节点类型编码
        type_code = self.node_type_map[self.language].get(node_type, self.node_type_map[self.language]["Other"])
        features = [type_code / len(self.node_type_map[self.language])]  # 归一化

        # 扩展特征：语义/结构特征（通用）
        if hasattr(node, "id") and isinstance(getattr(node, "id"), str):
            features.append(len(getattr(node, "id")) / 32)  # 名称长度（归一化到32）
            features.append(len(re.findall(r'[a-zA-Z]', getattr(node, "id"))) / 32)  # 字母数
        elif hasattr(node, "name") and isinstance(getattr(node, "name"), str):
            features.append(len(getattr(node, "name")) / 32)
            features.append(len(re.findall(r'[a-zA-Z]', getattr(node, "name"))) / 32)
        else:
            features.extend([0.0, 0.0])

        # 填充到指定维度
        while len(features) < self.node_feature_dim:
            features.append(0.0)

        return features[:self.node_feature_dim]  # 截断（防止超维度）

    def _traverse_python_ast(self, node: ast.AST, parent_id: Optional[int] = None):
        """Python AST遍历（通用逻辑）"""
        if self.language != "python":
            return

        node_type = type(node).__name__
        # 添加当前节点
        self.nodes.append(self._extract_node_features(node, node_type))
        self.node_metadata[self.node_id] = node_type
        current_id = self.node_id
        self.node_id += 1

        # 添加父边（无向）
        if parent_id is not None:
            self.edges.append([parent_id, current_id])
            self.edges.append([current_id, parent_id])

        # 递归遍历子节点
        for child in ast.iter_child_nodes(node):
            self._traverse_python_ast(child, current_id)

    def build_graph(self, code_snippet: str) -> Data:
        """
        通用图构建接口（对外暴露，与数据集无关）
        :param code_snippet: 任意代码字符串（Python/Java）
        :return: PyG Data对象（x: 节点特征, edge_index: 边索引, metadata: 节点元信息）
        """
        self.reset()

        try:
            if self.language == "python" and self.graph_type == "ast":
                tree = ast.parse(code_snippet)
                self._traverse_python_ast(tree)
            # 可扩展：Java AST解析、CFG构建等
            # elif self.language == "java" and self.graph_type == "ast":
            #     self._traverse_java_ast(...)
        except (SyntaxError, Exception) as e:
            # 代码解析失败时返回默认图（避免崩溃）
            self.nodes.append(self._extract_node_features("", "Other"))
            self.edges.append([0, 0])

        # 转换为PyTorch张量（通用格式）
        x = torch.tensor(self.nodes, dtype=torch.float32)
        edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous() if self.edges else torch.tensor(
            [[0], [0]])

        # 返回通用Data对象（携带元信息，便于调试）
        return Data(
            x=x,
            edge_index=edge_index,
            node_metadata=self.node_metadata,
            language=self.language,
            graph_type=self.graph_type
        )