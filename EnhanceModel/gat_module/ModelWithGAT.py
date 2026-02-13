import ast
import torch
import warnings
from transformers import AutoTokenizer
from typing import Tuple, Dict, List, Optional


# ========== 辅助工具：AST节点位置匹配token ==========
class CodeASTAnalyzer:
    """解析代码AST，提取节点、位置和关系"""

    def __init__(self, code: str, tokenizer: AutoTokenizer):
        self.code = code
        self.tokenizer = tokenizer
        self.tree: Optional[ast.AST] = None
        self.node_list: List[ast.AST] = []  # 所有AST节点
        self.node_type_map: Dict[int, str] = {}  # 节点idx -> 节点类型（如FunctionDef、Arg）
        self.node_pos_map: Dict[int, Tuple[int, int]] = {}  # 节点idx -> (起始字符位置, 结束字符位置)
        self.parent_child_map: Dict[int, List[int]] = {}  # 父节点idx -> 子节点idx列表

        # 解析AST
        self._parse_ast()
        # 提取节点信息和关系
        self._extract_node_info()

    def _parse_ast(self):
        """解析代码为AST树，处理解析错误"""
        try:
            self.tree = ast.parse(self.code)
        except SyntaxError as e:
            warnings.warn(f"代码语法错误，降级为模拟模式: {e}")
            self.tree = None
        except Exception as e:
            warnings.warn(f"AST解析失败，降级为模拟模式: {e}")
            self.tree = None

    def _extract_node_info(self):
        """递归提取AST节点的位置和父子关系"""
        if self.tree is None:
            return

        # 递归遍历AST
        def traverse(node: ast.AST, parent_idx: int = -1):
            # 记录当前节点
            node_idx = len(self.node_list)
            self.node_list.append(node)
            self.node_type_map[node_idx] = node.__class__.__name__

            # 记录节点的字符位置（尽可能精确）
            if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
                # 计算字符位置（简化版：按行和列估算）
                lines = self.code.split('\n')
                if node.lineno - 1 < len(lines):
                    line_start = sum(len(l) + 1 for l in lines[:node.lineno - 1])
                    start_pos = line_start + node.col_offset
                    # 估算结束位置（取节点内容的末尾）
                    end_pos = start_pos + len(ast.unparse(node)) if hasattr(ast, 'unparse') else start_pos + 10
                    self.node_pos_map[node_idx] = (start_pos, end_pos)

            # 记录父子关系
            if parent_idx != -1:
                if parent_idx not in self.parent_child_map:
                    self.parent_child_map[parent_idx] = []
                self.parent_child_map[parent_idx].append(node_idx)

            # 递归处理子节点
            for child in ast.iter_child_nodes(node):
                traverse(child, node_idx)

        traverse(self.tree)

    def get_token_node_mapping(self, token_positions: List[Tuple[int, int]]) -> Dict[int, int]:
        """
        匹配token的字符位置和AST节点位置，返回：token_idx -> node_idx
        token_positions: 每个token的(起始字符位置, 结束字符位置)
        """
        token_node_map = {}
        for token_idx, (token_start, token_end) in enumerate(token_positions):
            # 找包含该token的节点（优先匹配最细粒度节点）
            matched_node_idx = 0  # 默认匹配根节点
            min_node_length = float('inf')

            for node_idx, (node_start, node_end) in self.node_pos_map.items():
                if token_start >= node_start and token_end <= node_end:
                    node_length = node_end - node_start
                    if node_length < min_node_length:
                        min_node_length = node_length
                        matched_node_idx = node_idx

            token_node_map[token_idx] = matched_node_idx
        return token_node_map


# ========== 真实版：构建代码图结构 ==========
def build_code_graph(
        code: str,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        min_nodes: int = 1  # 最少节点数（防止无节点）
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    真实版：基于AST解析代码，构建真实的图结构
    返回：input_ids (1, L), adj_matrix (1, N, N), align_matrix (1, L, N)
    """
    # ========== 1. 编码代码为token，并记录token的字符位置 ==========
    # 带字符位置的编码（关键：匹配token和AST节点）
    encoding = tokenizer(
        code,
        return_tensors='pt',
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True  # 关键：返回token的字符位置
    )
    input_ids = encoding.input_ids
    offset_mapping = encoding.offset_mapping[0].tolist()  # (token_idx, (start, end))
    seq_len = input_ids.shape[1]

    # 过滤padding的token位置
    token_positions = []
    for start, end in offset_mapping:
        if start == 0 and end == 0:  # padding token
            token_positions.append((-1, -1))
        else:
            token_positions.append((start, end))

    # ========== 2. 解析AST并提取节点/关系 ==========
    analyzer = CodeASTAnalyzer(code, tokenizer)
    num_nodes = max(len(analyzer.node_list), min_nodes)

    # 降级逻辑：AST解析失败则用模拟版
    if analyzer.tree is None or num_nodes == 0:
        return _build_simulation_graph(code, tokenizer, max_length)

    # ========== 3. 构建真实邻接矩阵（基于AST父子关系） ==========
    adj_matrix = torch.zeros(1, num_nodes, num_nodes)
    # 填充父子节点连接（无向图）
    for parent_idx, child_indices in analyzer.parent_child_map.items():
        for child_idx in child_indices:
            if parent_idx < num_nodes and child_idx < num_nodes:
                adj_matrix[0, parent_idx, child_idx] = 1
                adj_matrix[0, child_idx, parent_idx] = 1  # 无向图

    # 确保至少有自环（防止孤立节点）
    for i in range(num_nodes):
        adj_matrix[0, i, i] = 1

    # ========== 4. 构建真实token-节点对齐矩阵 ==========
    align_matrix = torch.zeros(1, seq_len, num_nodes)
    # 匹配token和节点
    token_node_map = analyzer.get_token_node_mapping(token_positions)

    for token_idx in range(seq_len):
        node_idx = token_node_map.get(token_idx, 0)
        if node_idx < num_nodes:
            align_matrix[0, token_idx, node_idx] = 1  # 该token对应这个节点

    return input_ids, adj_matrix, align_matrix


# ========== 降级模拟版（兼容错误场景） ==========
def _build_simulation_graph(
        code: str,
        tokenizer: AutoTokenizer,
        max_length: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AST解析失败时的降级方案（和原模拟版一致）"""
    encoding = tokenizer(
        code,
        return_tensors='pt',
        padding='max_length',
        max_length=max_length,
        truncation=True
    )
    input_ids = encoding.input_ids
    seq_len = input_ids.shape[1]
    num_nodes = 5

    # 模拟邻接矩阵
    adj_matrix = torch.zeros(1, num_nodes, num_nodes)
    adj_matrix[0, 0, 1] = 1
    adj_matrix[0, 1, 2] = 1
    adj_matrix[0, 2, 3] = 1
    adj_matrix[0, 3, 4] = 1
    adj_matrix[0, 0, 4] = 1
    adj_matrix = adj_matrix + adj_matrix.transpose(1, 2)  # 无向图

    # 模拟对齐矩阵
    align_matrix = torch.zeros(1, seq_len, num_nodes)
    for i in range(seq_len):
        align_matrix[0, i, i % num_nodes] = 1

    return input_ids, adj_matrix, align_matrix