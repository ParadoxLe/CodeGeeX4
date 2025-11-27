# 使用 GNN 进行候选解重排序
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import List, Tuple


# ================= 1. GNN 模型定义 =================
class CodeGCN(torch.nn.Module):
    """
    一个简单的图卷积网络，用于对代码图进行评分。
    输入：节点特征（Embedding），边索引
    输出：0-1 之间的标量（代码质量分数）
    """

    def __init__(self, num_node_types, embedding_dim=64, hidden_dim=32):
        super(CodeGCN, self).__init__()
        # 节点嵌入层：将 AST 节点类型映射为向量
        self.embedding = nn.Embedding(num_node_types, embedding_dim)

        # 图卷积层
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # 评分头（全连接层）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 输出概率
        )

    def forward(self, x, edge_index, batch):
        # x: [num_nodes, 1] -> [num_nodes, embedding_dim]
        x = self.embedding(x.squeeze())

        # GCN 消息传递
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 全局池化：将节点特征聚合为整个图的特征向量
        # batch 向量指示了每个节点属于哪个图
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]

        # 评分
        out = self.classifier(x)
        return out


# ================= 2. AST 转图工具 =================
class ASTGraphBuilder(ast.NodeVisitor):
    """
    将 Python 代码解析为 PyG 所需的节点和边数据
    """

    def __init__(self):
        self.nodes = []  # 存储节点类型 ID
        self.edges = []  # 存储 (source, target)
        self.node_count = 0
        self.parent_stack = []  # 用于追踪父节点

        # 简单的 AST 类型映射表 (仅作示例，实际可更丰富)
        self.type_map = {
            'FunctionDef': 1, 'AsyncFunctionDef': 2, 'ClassDef': 3, 'Return': 4,
            'Delete': 5, 'Assign': 6, 'AugAssign': 7, 'AnnAssign': 8, 'For': 9,
            'AsyncFor': 10, 'While': 11, 'If': 12, 'With': 13, 'AsyncWith': 14,
            'Raise': 15, 'Try': 16, 'Assert': 17, 'Import': 18, 'ImportFrom': 19,
            'Global': 20, 'Nonlocal': 21, 'Expr': 22, 'Pass': 23, 'Break': 24,
            'Continue': 25, 'Call': 26, 'Name': 27, 'Constant': 28, 'arg': 29
        }
        self.default_type_id = 0

    def _get_type_id(self, node):
        return self.type_map.get(type(node).__name__, self.default_type_id)

    def generic_visit(self, node):
        # 创建当前节点
        node_id = self.node_count
        type_id = self._get_type_id(node)
        self.nodes.append(type_id)
        self.node_count += 1

        # 如果有父节点，建立边 (父 -> 子)
        if self.parent_stack:
            parent_id = self.parent_stack[-1]
            self.edges.append([parent_id, node_id])
            self.edges.append([node_id, parent_id])  # 双向边，利于信息流动

        # 压栈，访问子节点
        self.parent_stack.append(node_id)
        super().generic_visit(node)
        self.parent_stack.pop()


def code_to_graph(code: str) -> Data:
    """
    将代码字符串转换为 PyTorch Geometric Data 对象
    """
    try:
        tree = ast.parse(code)
        builder = ASTGraphBuilder()
        builder.visit(tree)

        if builder.node_count == 0:
            return None

        x = torch.tensor(builder.nodes, dtype=torch.long).unsqueeze(1)
        edge_index = torch.tensor(builder.edges, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)
    except Exception:
        # 解析失败（如语法错误），返回 None
        return None


# ================= 3. 对外接口类 =================
class GNNReranker:
    def __init__(self, model_path: str = None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # 初始化模型参数
        self.num_node_types = 50  # 根据 AST 映射表大小设定
        self.model = CodeGCN(num_node_types=self.num_node_types).to(self.device)

        if model_path:
            try:
                # 尝试加载预训练权重
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f" GNN Reranker model loaded from {model_path}")
                self.model.eval()
            except FileNotFoundError:
                print(f" Warning: GNN model path {model_path} not found. Using random weights (TESTING MODE).")
                self.model.eval()
        else:
            print(" GNN Reranker initialized with random weights (For pipeline testing only).")
            self.model.eval()

    def rerank(self, candidates: List[str]) -> List[str]:
        """
        输入候选代码列表，返回按 GNN 分数排序后的列表
        """
        graph_list = []
        valid_indices = []

        # 1. 转换图数据
        for idx, code in enumerate(candidates):
            graph = code_to_graph(code)
            if graph is not None:
                graph_list.append(graph)
                valid_indices.append(idx)
            # 解析失败的代码将被视为低分，稍后处理

        if not graph_list:
            return candidates  # 如果所有代码都无法解析，原样返回

        # 2. 批量推理
        batch_data = Batch.from_data_list(graph_list).to(self.device)

        with torch.no_grad():
            scores = self.model(batch_data.x, batch_data.edge_index, batch_data.batch)
            scores = scores.cpu().view(-1).tolist()  # 转为 list

        # 3. 组合结果 (score, original_code)
        scored_candidates = []

        # 处理能被 GNN 评分的代码
        for i, score in enumerate(scores):
            original_idx = valid_indices[i]
            scored_candidates.append((score, candidates[original_idx]))

        # 处理无法被 GNN 评分的代码（解析失败的），给 -1 分排在最后
        all_indices = set(range(len(candidates)))
        failed_indices = all_indices - set(valid_indices)
        for idx in failed_indices:
            scored_candidates.append((-1.0, candidates[idx]))

        # 4. 排序（分数高在前）
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # 打印调试信息（可选）
        # print(f"  GNN Top score: {scored_candidates[0][0]:.4f}")

        return [item[1] for item in scored_candidates]


# 自测代码
if __name__ == "__main__":
    reranker = GNNReranker()
    sample_codes = [
        "def add(a, b): return a + b",
        "def add(a, b): return a - b",  # 错误逻辑
        "def add(a, b): if a: return a else: return b"
    ]
    ranked = reranker.rerank(sample_codes)
    print("Original:", sample_codes)
    print("Ranked:", ranked)