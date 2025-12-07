import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import tree_sitter
import tree_sitter_python as tspython
from typing import List, Tuple, Optional, Dict

# -------------------------- DFG提取器 --------------------------
class DFGExtractor:
    """Python代码数据流图（DFG）提取工具"""
    def __init__(self):
        # 初始化Tree-sitter Python解析器
        self.lang = tree_sitter.Language(tspython.language())
        self.parser = tree_sitter.Parser()

    def _traverse_ast(self, node: tree_sitter.Node, code: str) -> Tuple[List[Dict], List[Tuple[int, int]]]:
        """遍历AST提取DFG节点和边（内部辅助方法）"""
        nodes = []
        edges = []
        node_id_map = {}  # 去重：变量/函数名 → 节点ID
        current_id = 0

        def add_node(node_type: str, value: str) -> int:
            """添加节点（去重），返回节点ID"""
            nonlocal current_id
            if value not in node_id_map:
                node_id_map[value] = current_id
                nodes.append({"id": current_id, "type": node_type, "value": value})
                current_id += 1
            return node_id_map[value]

        # 递归遍历AST，聚焦核心数据流
        def recurse(n: tree_sitter.Node):
            # 变量引用（数据节点）
            if n.type == "identifier":
                var_name = code[n.start_byte:n.end_byte].strip()
                if var_name:
                    add_node("data", var_name)
            # 赋值语句（a = b / func()）
            elif n.type == "assignment":
                target_node = n.child_by_field_name("left")
                value_node = n.child_by_field_name("right")
                if target_node and target_node.type == "identifier" and value_node:
                    target_var = code[target_node.start_byte:target_node.end_byte].strip()
                    target_id = add_node("data", target_var)
                    # 处理右侧变量
                    if value_node.type == "identifier":
                        src_var = code[value_node.start_byte:value_node.end_byte].strip()
                        src_id = add_node("data", src_var)
                        edges.append((src_id, target_id))
                    # 处理右侧函数调用
                    elif value_node.type == "call":
                        func_node = value_node.child_by_field_name("function")
                        if func_node and func_node.type == "identifier":
                            func_name = code[func_node.start_byte:func_node.end_byte].strip()
                            func_id = add_node("operation", func_name)
                            # 函数参数依赖
                            for arg in value_node.children:
                                if arg.type == "identifier":
                                    arg_var = code[arg.start_byte:arg.end_byte].strip()
                                    arg_id = add_node("data", arg_var)
                                    edges.append((arg_id, func_id))
                            # 函数输出 → 目标变量
                            edges.append((func_id, target_id))
            # 独立函数调用（无赋值）
            elif n.type == "call" and not (n.parent and n.parent.type == "assignment"):
                func_node = n.child_by_field_name("function")
                if func_node and func_node.type == "identifier":
                    func_name = code[func_node.start_byte:func_node.end_byte].strip()
                    func_id = add_node("operation", func_name)
                    # 函数参数依赖
                    for arg in n.children:
                        if arg.type == "identifier":
                            arg_var = code[arg.start_byte:arg.end_byte].strip()
                            arg_id = add_node("data", arg_var)
                            edges.append((arg_id, func_id))
            # 递归子节点
            for child in n.children:
                recurse(child)

        recurse(node)
        return nodes, edges

    def extract(self, code_snippet: str) -> Tuple[List[Dict], List[Tuple[int, int]]]:
        """对外接口：输入代码片段，输出DFG节点列表和边列表"""
        if not code_snippet.strip():
            return [], []
        tree = self.parser.parse(bytes(code_snippet, "utf8"))
        return self._traverse_ast(tree.root_node, code_snippet)

# -------------------------- 轻量GGNN增强器 --------------------------
class LightGGNNEnhancer(nn.Module):
    """轻量GGNN：从DFG提炼结构化数据流逻辑"""
    def __init__(self, input_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        # 2层GAT（聚焦关键节点，快速推理）
        self.gat1 = GATConv(input_dim, hidden_dim, heads=2, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        # 设备自动适配（CPU/GPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _encode_nodes(self, nodes: List[Dict]) -> torch.Tensor:
        """节点特征编码（64维向量）"""
        node_features = []
        for node in nodes:
            # 类型编码（data=0.0, operation=1.0）
            type_feat = torch.tensor([0.0]) if node["type"] == "data" else torch.tensor([1.0])
            # 字符串哈希编码（63维，保证输入维度统一）
            str_val = node["value"].strip()
            str_hash = torch.tensor([hash(str_val) % 1000 / 1000], dtype=torch.float32)
            str_feat = torch.cat([str_hash.repeat(63)], dim=0)
            # 拼接为64维特征
            feat = torch.cat([type_feat, str_feat], dim=0).to(self.device)
            node_features.append(feat)
        return torch.stack(node_features, dim=0)  # (num_nodes, 64)

    def get_dataflow_logic(self, code_snippet: str) -> str:
        """对外统一接口：输入代码片段，输出结构化数据流逻辑（自然语言）"""
        # 1. 提取DFG
        dfg_extractor = DFGExtractor()
        nodes, edges = dfg_extractor.extract(code_snippet)
        if not nodes or not edges:
            return ""

        # 2. 构建图数据
        x = self._encode_nodes(nodes)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
        graph_data = Data(x=x, edge_index=edge_index)

        # 3. GGNN推理（仅前向传播，不训练）
        with torch.no_grad():
            x = self.dropout(graph_data.x)
            x = self.relu(self.gat1(x, graph_data.edge_index))
            x = self.dropout(x)
            node_embeds = self.gat2(x, graph_data.edge_index)

        # 4. 提取核心数据流逻辑
        node_importance = torch.norm(node_embeds, dim=1)
        top_k = min(5, len(nodes))  # 取Top5核心节点
        top_node_ids = torch.topk(node_importance, k=top_k).indices.tolist()
        top_node_map = {node["id"]: node for node in nodes if node["id"] in top_node_ids}

        # 构建逻辑描述
        logic_descriptions = []
        for src_id, tgt_id in edges:
            if src_id in top_node_map and tgt_id in top_node_map:
                src_node = top_node_map[src_id]
                tgt_node = top_node_map[tgt_id]
                if src_node["type"] == "data" and tgt_node["type"] == "operation":
                    logic_descriptions.append(f"数据「{src_node['value']}」作为输入传给操作「{tgt_node['value']}」")
                elif src_node["type"] == "operation" and tgt_node["type"] == "data":
                    logic_descriptions.append(f"操作「{src_node['value']}」的输出为数据「{tgt_node['value']}」")
                elif src_node["type"] == "data" and tgt_node["type"] == "data":
                    logic_descriptions.append(f"数据「{src_node['value']}」赋值给数据「{tgt_node['value']}」")

        # 去重并格式化输出
        unique_logic = list(dict.fromkeys(logic_descriptions))  # 去重
        if unique_logic:
            return "请遵循以下数据流逻辑生成代码：" + "；".join(unique_logic) + "。"
        return ""

# -------------------------- 快捷实例（方便外部直接引用） --------------------------
# 初始化增强器（全局单例，避免重复创建）
enhancer = LightGGNNEnhancer()

# 对外暴露的核心函数（一行调用）
def get_code_enhance_prompt(code_snippet: str) -> str:
    """输入代码片段，返回DFG+GGNN增强后的提示词片段"""
    return enhancer.get_dataflow_logic(code_snippet)