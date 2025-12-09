# MBPP+/mbpp_valid.py（你的原有文件，新增/修改部分标注）
import json
import sys
import torch  # 新增

sys.path.append("..")  # 新增：让脚本能找到Model_Enhancer

# 新增：导入GAT通用模块
from Model_Enhancer.gnn_enhance import GenericGAT, GenericCodeGraphBuilder, MBPPGATAdapter

# 新增：初始化GAT（全局唯一，避免重复初始化）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAT_MODEL = GenericGAT(in_channels=8, hidden_channels=16, out_channels=1, task_type="regression").to(DEVICE)
GRAPH_BUILDER = GenericCodeGraphBuilder(language="python", graph_type="ast", node_feature_dim=8)
MBPP_ADAPTER = MBPPGATAdapter(GAT_MODEL, GRAPH_BUILDER)


# 你的原有验证函数（保持不变）
def verify_code(code, test_cases):
    """你原有代码：执行代码+测试用例，返回pass/fail+错误类型"""
    try:
        exec_globals = {}
        exec(code, exec_globals)
        for test in test_cases:
            exec(test, exec_globals)
        return {"result": "pass", "error_type": None}
    except SyntaxError as e:
        return {"result": "fail", "error_type": "syntax_error", "msg": str(e)}
    except Exception as e:
        return {"result": "fail", "error_type": "execution_error", "msg": str(e)}


# 新增：集成GAT的MBPP+验证函数（复用你的原有verify_code）
def verify_mbpp_plus_with_gat(mbpp_item):
    """
    输入：MBPP+单条数据（dict）
    输出：包含GAT得分的验证结果
    """
    # 1. 原有验证逻辑（完全复用你的代码）
    code = mbpp_item["generated_code"]  # 若你字段名是code，改为mbpp_item["code"]
    test_cases = mbpp_item["test_cases"]
    verify_result = verify_code(code, test_cases)

    # 2. GAT分析（新增，调用通用模块）
    gat_result = MBPP_ADAPTER.analyze_mbpp_code(mbpp_item)

    # 3. 合并结果（原有验证结果 + GAT得分）
    final_result = {
        "task_id": mbpp_item["task_id"],
        "gat_score": gat_result["code_quality_score"],  # GAT预测的代码质量得分
        "validation_result": verify_result["result"],
        "error_type": verify_result["error_type"],
        "raw_code": code
    }
    return final_result


# 你的原有主函数（可修改为调用新增的verify_mbpp_plus_with_gat）
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mbpp_valid.py <mbpp_plus_data_path>")
        sys.exit(1)
    data_path = sys.argv[1]

    # 读取MBPP+数据
    with open(data_path, "r", encoding="utf-8") as f:
        mbpp_plus_data = [json.loads(line) for line in f]

    # 批量验证（集成GAT）
    results = []
    for item in mbpp_plus_data:
        res = verify_mbpp_plus_with_gat(item)
        results.append(res)
        print(f"Task {item['task_id']}: GAT Score={res['gat_score']:.4f}, Result={res['validation_result']}")

    # 保存结果
    with open("mbpp_plus_result_with_gat.jsonl", "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    # 统计效果（新增：分析GAT与验证结果的相关性）
    total = len(results)
    pass_num = sum(1 for r in results if r["validation_result"] == "pass")
    pass_rate = pass_num / total

    # GAT得分阈值分析（比如得分>0.5的代码通过率）
    high_gat_pass = sum(1 for r in results if r["gat_score"] > 0.5 and r["validation_result"] == "pass")
    high_gat_total = sum(1 for r in results if r["gat_score"] > 0.5)
    high_gat_pass_rate = high_gat_pass / high_gat_total if high_gat_total > 0 else 0.0

    print(f"\n===== MBPP+ 评估结果 =====")
    print(f"总任务数: {total}")
    print(f"整体通过率: {pass_rate:.4f}")
    print(f"GAT高得分(>0.5)任务数: {high_gat_total}")
    print(f"GAT高得分任务通过率: {high_gat_pass_rate:.4f}")