import json
import os
import warnings
from datetime import datetime
# 屏蔽HuggingFace的FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# 设置镜像源（需在加载模型前执行）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from typing import Tuple


# 保留基础的读取、写入函数（和MBPP验证代码完全一致）
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [json.loads(line.strip()) for line in file]
    return lines


def write_jsonl(data, file_path):
    # 确保文件所在目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')


def write_summary_json(summary_dict, file_path):
    """专门写入验证统计汇总信息（JSON格式）"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=4)
    print(f"\n统计汇总已保存到：{os.path.abspath(file_path)}")


# 核心验证函数（适配HumanEval：增加入口函数检查）
def run_tests(function_code: str, test_cases: list, entry_point: str) -> Tuple[bool, str]:
    """
    运行HumanEval测试用例验证代码正确性，返回(是否通过, 错误信息)
    新增 entry_point 参数：HumanEval每个题目要求实现的目标函数名
    """
    exec_env = {}
    # 执行生成的函数代码
    try:
        exec(function_code, exec_env)
    except SyntaxError as e:
        return False, f"语法错误: {str(e)[:100]}"  # 截取前100字符避免输出过长
    except Exception as e:
        return False, f"代码执行错误: {str(e)[:100]}"

    # 关键适配：检查目标入口函数是否存在（HumanEval核心要求）
    if entry_point not in exec_env:
        return False, f"入口函数缺失: 未找到要求实现的函数「{entry_point}」"

    # 执行所有测试用例
    for idx, test_case in enumerate(test_cases):
        try:
            exec(test_case, exec_env)  # 执行assert测试用例
        except AssertionError:
            return False, f"测试用例{idx + 1}失败: {test_case[:80]}..."  # 截取测试用例避免过长
        except Exception as e:
            return False, f"测试用例{idx + 1}运行错误: {str(e)[:100]} (用例: {test_case[:50]}...)"

    return True, "所有测试用例通过"


# 主函数：统计结果保存为JSON格式（适配HumanEval数据集特性）
def process_and_validate(file_path1, output_dir="humaneval_validation_results", humaneval_dataset_path="evalplus/humanevalplus"):
    # 1. 创建结果文件夹（不存在则自动创建）
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存到文件夹：{os.path.abspath(output_dir)}")

    # 2. 读取清洗后的代码文件和HumanEval数据集
    generated_data = read_jsonl(file_path1)
    humaneval = load_dataset(humaneval_dataset_path)
    test_dataset = humaneval["test"]

    # 提取数据集信息（用于汇总）
    dataset_task_indices = [idx for idx in range(len(test_dataset))]  # HumanEval用索引匹配，无task_id字段
    dataset_total = len(test_dataset)
    dataset_id_min = min(dataset_task_indices) if dataset_task_indices else None
    dataset_id_max = max(dataset_task_indices) if dataset_task_indices else None

    print(f"\n=== HumanEval 数据集信息 ===")
    print(f"test分割总题目数：{dataset_total}")
    print(f"题目索引范围：{dataset_id_min} - {dataset_id_max}")
    print(f"前5个题目入口函数示例：{[test_dataset[idx]['entry_point'] for idx in range(5)]}")
    print(f"========================\n")

    # 构建：题目索引 → 测试用例+入口函数 的映射（HumanEval核心适配）
    idx_to_test_info = {
        idx: {
            "test_cases": item["test"],
            "entry_point": item["entry_point"]
        } for idx, item in enumerate(test_dataset)
    }

    updated_data = []  # 所有题目的完整结果
    total_passed = 0
    total = len(generated_data)
    unknown_idx_count = 0
    error_stats = {  # 适配HumanEval，新增「入口函数缺失」错误类型
        "task_id格式错误": 0,
        "未知题目索引": 0,
        "无有效代码": 0,
        "语法错误": 0,
        "代码执行错误": 0,
        "入口函数缺失": 0,
        "测试用例失败": 0
    }

    for entry in generated_data:
        new_entry = entry.copy()
        original_task_id = entry["task_id"]  # HumanEval的task_id格式：HumanEval/0、HumanEval/1...

        # 处理task_id（适配HumanEval：提取索引，去掉"HumanEval/"前缀）
        try:
            task_id_str = original_task_id.lower().replace("humaneval/", "").strip()
            task_idx = int(task_id_str)  # HumanEval用整数索引匹配题目
        except ValueError:
            new_entry['validation_result'] = f"task_id格式错误（无法转为整数索引）：{original_task_id}"
            new_entry['passed'] = False
            error_stats["task_id格式错误"] += 1
            updated_data.append(new_entry)
            print(f"处理题目 {original_task_id} → 失败: {new_entry['validation_result']}")
            continue

        # 读取清洗后的纯代码
        cleaned_code = entry['solution'].strip()
        new_entry['cleaned_solution'] = cleaned_code

        # 验证逻辑（适配HumanEval：用索引匹配题目，检查入口函数）
        if task_idx not in idx_to_test_info:
            new_entry['validation_result'] = f"未知题目索引（数据集无该题目）：{task_idx}"
            new_entry['passed'] = False
            unknown_idx_count += 1
            error_stats["未知题目索引"] += 1
        else:
            test_info = idx_to_test_info[task_idx]
            test_cases = test_info["test_cases"]
            entry_point = test_info["entry_point"]  # 目标函数名
            if not cleaned_code:
                new_entry['validation_result'] = "无有效代码"
                new_entry['passed'] = False
                error_stats["无有效代码"] += 1
            else:
                # 调用验证函数：传入入口函数名（HumanEval核心适配）
                passed, msg = run_tests(cleaned_code, test_cases, entry_point)
                new_entry['validation_result'] = msg
                new_entry['passed'] = passed
                if passed:
                    total_passed += 1
                else:
                    # 统计具体错误类型（适配新增的「入口函数缺失」）
                    if "语法错误" in msg:
                        error_stats["语法错误"] += 1
                    elif "代码执行错误" in msg:
                        error_stats["代码执行错误"] += 1
                    elif "入口函数缺失" in msg:
                        error_stats["入口函数缺失"] += 1
                    elif "测试用例" in msg:
                        error_stats["测试用例失败"] += 1

        updated_data.append(new_entry)
        # 实时打印进度
        print(
            f"处理题目 {original_task_id} → {'通过' if new_entry['passed'] else '失败'}: {new_entry['validation_result'][:100]}")

    # 3. 计算核心统计信息
    pass_rate = (total_passed / total) * 100 if total > 0 else 0.0
    failed_count = total - total_passed  # 总失败数

    # 构造结构化的汇总字典（JSON格式）
    summary_dict = {
        "验证状态": "完成",
        "验证时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "核心统计": {
            "总题目数": total,
            "通过题目数": total_passed,
            "失败题目数": failed_count,
            "未知索引题目数": unknown_idx_count,
            "Pass@1 通过率(%)": round(pass_rate, 2)
        },
        "错误类型统计": error_stats,
        "数据集信息": {
            "数据集名称": humaneval_dataset_path,
            "分割名称": "test",
            "数据集总题目数": dataset_total,
            "题目索引范围": f"{dataset_id_min} - {dataset_id_max}"
        },
        "文件信息": {
            "输入文件路径": os.path.abspath(file_path1),
            "完整结果文件路径": os.path.join(output_dir, "codegeex4-all-9b-HumanEval-validated-full.jsonl"),
            "统计汇总文件路径": os.path.join(output_dir, "humaneval_validation_summary.json")
        }
    }

    # 4. 定义输出文件路径（放入结果文件夹）
    full_result_path = os.path.join(output_dir, "codegeex4-all-9b-HumanEval-validated-full.jsonl")
    summary_path = os.path.join(output_dir, "humaneval_validation_summary.json")

    # 5. 保存结果
    write_jsonl(updated_data, full_result_path)
    write_summary_json(summary_dict, summary_path)

    # 6. 终端打印汇总信息
    print(f"\n=== 验证完成 ===")
    print(f"总题目数: {total}")
    print(f"通过题目数: {total_passed}")
    print(f"失败题目数: {failed_count}")
    print(f"未知索引题目数: {unknown_idx_count}")
    print(f"Pass@1 通过率: {pass_rate:.2f}%")
    print(f"\n错误类型统计：")
    for err_type, count in error_stats.items():
        print(f"- {err_type}: {count}个")
    print(f"\n文件保存位置：")
    print(f"- 完整结果（所有题目）: {os.path.abspath(full_result_path)}")
    print(f"- 统计汇总（JSON）: {os.path.abspath(summary_path)}")


# ---------------------- 执行配置（适配HumanEval文件） ----------------------
# 输入文件：HumanEval清洗后的纯代码文件
file_path1 = 'codegeex4-all-9b_HumanEval-sanitized.jsonl'
# 输出文件夹：存放HumanEval验证结果（自动创建）
output_dir = "humaneval_validation_results"

# 执行完整流程
process_and_validate(file_path1, output_dir)