import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from typing import Dict, List


def graph2nx(data: Data) -> nx.Graph:
    """通用转换：PyG Data → NetworkX图（可视化/分析）"""
    G = nx.Graph()
    # 添加节点
    for node_id in range(data.x.shape[0]):
        G.add_node(node_id, feature=data.x[node_id].numpy(), type=data.node_metadata.get(node_id, "Unknown"))
    # 添加边
    for u, v in data.edge_index.t().numpy():
        G.add_edge(u, v)
    return G

def visualize_graph(data: Data, save_path: str = "code_graph.png"):
    """通用图可视化（与数据集无关）"""
    G = graph2nx(data)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): "" for u, v in G.edges()})
    plt.title(f"Code Graph ({data.language}/{data.graph_type})")
    plt.savefig(save_path)
    plt.close()

def batch_graphs(graphs: List[Data]) -> Dict[str, torch.Tensor]:
    """通用批处理：多个图打包为批次（提升推理效率）"""
    x_list, edge_index_list = [], []
    ptr = 0
    for g in graphs:
        x_list.append(g.x)
        edge_index_list.append(g.edge_index + ptr)
        ptr += g.x.shape[0]
    return {
        "x": torch.cat(x_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "ptr": ptr
    }