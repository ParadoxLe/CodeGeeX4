import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data, Batch
from tree_sitter_languages import get_language, get_parser


# ==========================================
# 1. 图构建工具 (AST -> Graph)
# ==========================================
class ASTGraphBuilder:
    def __init__(self):
        # 使用 tree_sitter_languages 自动管理语言包，无需手动编译
        self.language = get_language('python')
        self.parser = get_parser('python')

        # 简化的节点类型词表 (将AST节点类型映射为ID)
        # 实际训练时应构建完整的 vocab，这里使用简单的哈希模拟
        self.max_vocab_size = 200  # 常见的AST节点类型数量约在100-200之间

    def get_node_type_id(self, node_type: str) -> int:
        """简单的哈希映射，将节点类型字符串转为ID"""
        return abs(hash(node_type)) % self.max_vocab_size

    def code_to_graph(self, code: str) -> Data:
        """将代码字符串转换为 PyG Data 对象"""
        try:
            tree = self.parser.parse(bytes(code, "utf8"))
        except Exception as e:
            # 如果解析失败（如代码语法极度错误），返回空图
            return self._empty_graph()

        root_node = tree.root_node
        node_features = []
        edge_indices = []

        # 遍历 AST 构建节点和边
        # 使用栈进行深度优先遍历，同时记录 (node, parent_index)
        stack = [(root_node, -1)]
        current_node_index = 0

        while stack:
            node, parent_idx = stack.pop()

            # 1. 构建节点特征 (这里只用节点类型的 Embedding)
            type_id = self.get_node_type_id(node.type)
            node_features.append(type_id)

            # 2. 构建边 (双向边：父->子，子->父)
            if parent_idx != -1:
                edge_indices.append([parent_idx, current_node_index])
                edge_indices.append([current_node_index, parent_idx])

            # 将子节点压入栈
            # 注意：tree-sitter 的 children 属性
            for child in node.children:
                stack.append((child, current_node_index))

            current_node_index += 1

        if len(node_features) == 0:
            return self._empty_graph()

        # 转换为 Tensor
        x = torch.tensor(node_features, dtype=torch.long).unsqueeze(1)  # [Num_Nodes, 1]
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()  # [2, Num_Edges]

        if edge_index.numel() == 0:
            # 处理只有一个根节点没有边的情况
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    def _empty_graph(self):
        """返回一个占位用的空图"""
        x = torch.zeros((1, 1), dtype=torch.long)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)


# ==========================================
# 2. GIN 模型定义 (Graph Isomorphism Network)
# ==========================================
class GINCodeModel(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers=3):
        super(GINCodeModel, self).__init__()

        # 节点嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # GIN 的核心是一个 MLP (Linear -> ReLU -> Linear)
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            # train_eps=True 允许模型学习中心节点与邻居的权重比例
            self.convs.append(GINConv(mlp, train_eps=True))

        # 评分头 (输出 0-1 之间的分数)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, batch):
        # 1. Embedding
        # x 维度是 [Num_Nodes, 1]，需要去掉最后一维
        h = self.embedding(x.squeeze(-1))

        # 2. GIN Layers
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)

        # 3. Global Pooling (将节点特征聚合为图特征)
        # GIN 通常使用 add pooling 效果最好
        h_graph = global_add_pool(h, batch)

        # 4. Scoring
        score = self.classifier(h_graph)
        return score.view(-1)  # [Batch_Size]


# ==========================================
# 3. 重排序器主类 (供 main.py 调用)
# ==========================================
class GNNReranker:
    def __init__(self, model_path="gnn_model.pth", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graph_builder = ASTGraphBuilder()

        # 模型配置 (需与训练时保持一致)
        self.hidden_dim = 64
        self.vocab_size = self.graph_builder.max_vocab_size

        # 初始化模型
        self.model = GINCodeModel(self.vocab_size, self.hidden_dim).to(self.device)
        self.model.eval()

        # 加载权重
        if os.path.exists(model_path):
            print(f" [GNNReranker] Loading GIN weights from {model_path}")
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f" [GNNReranker] Error loading weights: {e}")
                print("  Running with RANDOM weights (Results won't be improved)")
        else:
            print(f" [GNNReranker] Weights file '{model_path}' not found.")
            print("  Running with RANDOM initialized GIN (Just for testing the pipeline).")
            print("  To get better results, please train the GNN on MBPP/APPS dataset first.")

    def rerank(self, candidates: list) -> list:
        """
        接收代码候选列表，返回按GNN分数排序后的列表
        """
        if not candidates:
            return []

        # 1. 将所有候选代码转换为图数据
        data_list = []
        valid_indices = []

        for i, code in enumerate(candidates):
            graph_data = self.graph_builder.code_to_graph(code)
            # 如果图节点太少（比如解析失败），可能导致计算问题，可以过滤或保留
            if graph_data.x.size(0) > 0:
                data_list.append(graph_data)
                valid_indices.append(i)

        if not data_list:
            return candidates  # 如果所有转换都失败，原样返回

        # 2. 创建 Batch (PyG 处理多个图的方式)
        batch_data = Batch.from_data_list(data_list).to(self.device)

        # 3. 推理评分
        with torch.no_grad():
            scores = self.model(batch_data.x, batch_data.edge_index, batch_data.batch)
            scores = scores.cpu().tolist()

        # 4. 组合 (候选代码, 分数) 并排序
        scored_candidates = []
        for i, score in enumerate(scores):
            original_idx = valid_indices[i]
            scored_candidates.append((candidates[original_idx], score))

        # 补充那些转换失败的（给最低分 -1.0）
        parsed_indices = set(valid_indices)
        for i, code in enumerate(candidates):
            if i not in parsed_indices:
                scored_candidates.append((code, -1.0))

        # 按分数从高到低排序
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # 返回纯代码列表
        sorted_codes = [item[0] for item in scored_candidates]

        # Debug 打印（可选）
        # if len(sorted_codes) > 0:
        #     print(f"    [GNN Debug] Top score: {scored_candidates[0][1]:.4f}, Low score: {scored_candidates[-1][1]:.4f}")

        return sorted_codes