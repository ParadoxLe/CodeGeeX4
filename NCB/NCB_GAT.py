# NCB_valid.py
from Model_Enhancer.gnn_enhance import GenericGAT, GenericCodeGraphBuilder
import torch

# 1. 适配Java语言（仅修改构建器参数）
gat = GenericGAT(in_channels=8, hidden_channels=16, out_channels=2, task_type="multiclass")
graph_builder = GenericCodeGraphBuilder(language="java", graph_type="ast", node_feature_dim=8)

# 2. 直接调用通用接口（无需适配器，极简）
def analyze_java_code(java_code):
    graph_data = graph_builder.build_graph(java_code)
    with torch.no_grad():
        # 多分类预测（错误类型：0=语法错误，1=逻辑错误）
        graph_feat = gat.aggregate_graph(graph_data.x, graph_data.edge_index, agg_type="max")
        error_type = torch.argmax(graph_feat).item()
    return {"error_type": error_type, "confidence": graph_feat[error_type].item()}

# 测试Java代码
java_code = "public class Test { public static int add(int a, int b) { return a + b; } }"
result = analyze_java_code(java_code)
print(f"NCB Java代码分析结果：{result}")