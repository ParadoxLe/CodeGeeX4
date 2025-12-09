import json
from Model_Enhancer.gnn_enhance import GenericGAT, GenericCodeGraphBuilder, HumanEvalGATAdapter

# 1. 复用通用模块（无需修改GAT/图构建器）
gat = GenericGAT(in_channels=8, hidden_channels=16, out_channels=1, task_type="regression")
graph_builder = GenericCodeGraphBuilder(language="python", graph_type="ast", node_feature_dim=8)

# 2. 加载HumanEval+适配器
humaneval_adapter = HumanEvalGATAdapter(gat, graph_builder)

# 3. 处理HumanEval+数据（完全复用核心逻辑）
def validate_humaneval_code(humaneval_item):
    gat_result = humaneval_adapter.analyze_humaneval_code(humaneval_item)
    # 原有HumanEval+验证逻辑...
    return {
        "task_id": humaneval_item["task_id"],
        "gat_score": gat_result["code_quality_score"],
        "validation_result": "fail"  # 原有验证结果
    }

if __name__ == "__main__":
    with open("humaneval_data.jsonl", "r") as f:
        humaneval_data = [json.loads(line) for line in f]
    results = [validate_humaneval_code(item) for item in humaneval_data]
    print(f"HumanEval+验证完成，共处理{len(results)}条数据")