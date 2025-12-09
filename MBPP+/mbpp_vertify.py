import json
import os
import warnings
from datetime import datetime  # 用于记录验证时间（无需额外安装）

# 屏蔽HuggingFace的FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# 设置镜像源（需在加载模型前执行）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from typing import Tuple


# 保留基础的读取、写入函数
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
        json.dump(summary_dict, f, ensure_ascii=False, indent=4)  # indent=4美化格式，方便阅读
    print(f"\n统计汇总已保存到：{os.path.abspath(file_path)}")


# 核心验证函数（不变）
def run_tests(function_code: str, test_cases: list) -> Tuple[bool, str]:
    """运行测试用例验证代码正确性，返回(是否通过, 错误信息)"""
    exec_env = {}
    # 执行生成的函数代码
    try:
        exec(function_code, exec_env)
    except SyntaxError as e:
        return False, f"语法错误: {str(e)}"
    except Exception as e:
        return False, f"代码执行错误: {str(e)}"

    # 执行所有测试用例
    for idx, test_case in enumerate(test_cases):
        try:
            exec(test_case, exec_env)  # 执行assert测试用例
        except AssertionError:
            return False, f"测试用例{idx + 1}失败: {test_case}"
        except Exception as e:
            return False, f"测试用例{idx + 1}运行错误: {str(e)} (用例: {test_case})"

    return True, "所有测试用例通过"


# 主函数：统计结果保存为JSON格式
def process_and_validate(file_path1, output_dir="mbpp_validation_results", mbpp_dataset_path="evalplus/mbppplus"):
    # 1. 创建结果文件夹（不存在则自动创建）
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存到文件夹：{os.path.abspath(output_dir)}")

    # 2. 读取清洗后的代码文件和MBPP+数据集
    generated_data = read_jsonl(file_path1)
    mbpp = load_dataset(mbpp_dataset_path)
    test_dataset = mbpp["test"]

    # 提取数据集信息（用于汇总）
    dataset_task_ids = [item["task_id"] for item in test_dataset]
    dataset_total = len(dataset_task_ids)
    dataset_id_min = min(dataset_task_ids) if dataset_task_ids else None
    dataset_id_max = max(dataset_task_ids) if dataset_task_ids else None

    print(f"\n=== MBPP+ 数据集信息 ===")
    print(f"test分割总题目数：{dataset_total}")
    print(f"task_id范围：{dataset_id_min} - {dataset_id_max}")
    print(f"前5个task_id示例：{dataset_task_ids[:5]}")
    print(f"========================\n")

    # 构建题目ID→测试用例映射（整数键）
    task_id_to_tests = {item["task_id"]: item["test_list"] for item in test_dataset}

    updated_data = []  # 所有题目的完整结果
    total_passed = 0
    total = len(generated_data)
    unknown_id_count = 0
    error_stats = {  # 新增：统计各类错误的数量（可选，丰富汇总信息）
        "task_id格式错误": 0,
        "未知题目ID": 0,
        "无有效代码": 0,
        "语法错误": 0,
        "代码执行错误": 0,
        "测试用例失败": 0
    }

    for entry in generated_data:
        new_entry = entry.copy()
        original_task_id = entry["task_id"]

        # 处理task_id（前缀+类型转换）
        try:
            task_id_str = original_task_id.lower().replace("mbpp/", "").strip()
            task_id = int(task_id_str)
        except ValueError:
            new_entry['validation_result'] = f"task_id格式错误（无法转为整数）：{original_task_id}"
            new_entry['passed'] = False
            error_stats["task_id格式错误"] += 1
            updated_data.append(new_entry)
            print(f"处理题目 {original_task_id} → 失败: {new_entry['validation_result']}")
            continue

        # 读取清洗后的纯代码
        cleaned_code = entry['solution'].strip()
        new_entry['cleaned_solution'] = cleaned_code

        # 验证逻辑
        if task_id not in task_id_to_tests:
            new_entry['validation_result'] = f"未知题目ID（数据集无该ID）：{task_id}"
            new_entry['passed'] = False
            unknown_id_count += 1
            error_stats["未知题目ID"] += 1
        else:
            test_cases = task_id_to_tests[task_id]
            if not cleaned_code:
                new_entry['validation_result'] = "无有效代码"
                new_entry['passed'] = False
                error_stats["无有效代码"] += 1
            else:
                passed, msg = run_tests(cleaned_code, test_cases)
                new_entry['validation_result'] = msg
                new_entry['passed'] = passed
                if passed:
                    total_passed += 1
                else:
                    # 统计具体错误类型（方便分析模型短板）
                    if "语法错误" in msg:
                        error_stats["语法错误"] += 1
                    elif "代码执行错误" in msg:
                        error_stats["代码执行错误"] += 1
                    elif "测试用例" in msg:
                        error_stats["测试用例失败"] += 1

        updated_data.append(new_entry)
        # 实时打印进度
        print(
            f"处理题目 {original_task_id} → {'通过' if new_entry['passed'] else '失败'}: {new_entry['validation_result']}")

    # 3. 计算核心统计信息
    pass_rate = (total_passed / total) * 100 if total > 0 else 0.0
    failed_count = total - total_passed  # 总失败数

    # 构造结构化的汇总字典（JSON格式）
    summary_dict = {
        "验证状态": "完成",
        "验证时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 记录当前时间
        "核心统计": {
            "总题目数": total,
            "通过题目数": total_passed,
            "失败题目数": failed_count,
            "未知ID题目数": unknown_id_count,
            "Pass@1 通过率(%)": round(pass_rate, 2)
        },
        "错误类型统计": error_stats,  # 各类错误的详细数量
        "数据集信息": {
            "数据集名称": mbpp_dataset_path,
            "分割名称": "test",
            "数据集总题目数": dataset_total,
            "数据集task_id范围": f"{dataset_id_min} - {dataset_id_max}"
        },
        "文件信息": {
            "输入文件路径": os.path.abspath(file_path1),
            "完整结果文件路径": os.path.join(output_dir, "codegeex4-all-9b-Mbpp+-validated-full.jsonl"),
            "统计汇总文件路径": os.path.join(output_dir, "validation_summary.json")
        }
    }

    # 4. 定义输出文件路径（放入结果文件夹）
    full_result_path = os.path.join(output_dir, "codegeex4-all-9b-Mbpp+-validated-full.jsonl")
    summary_path = os.path.join(output_dir, "validation_summary.json")  # 统计汇总（JSON格式）

    # 5. 保存结果（按需选择，不想保留完整结果可注释第一行）
    write_jsonl(updated_data, full_result_path)  # 所有题目的完整验证结果
    write_summary_json(summary_dict, summary_path)  # 核心统计汇总（JSON格式）

    # 6. 终端打印汇总信息（和JSON内容一致）
    print(f"\n=== 验证完成 ===")
    print(f"总题目数: {total}")
    print(f"通过题目数: {total_passed}")
    print(f"失败题目数: {failed_count}")
    print(f"未知ID题目数: {unknown_id_count}")
    print(f"Pass@1 通过率: {pass_rate:.2f}%")
    print(f"\n错误类型统计：")
    for err_type, count in error_stats.items():
        print(f"- {err_type}: {count}个")
    print(f"\n文件保存位置：")
    print(f"- 完整结果（所有题目）: {os.path.abspath(full_result_path)}")
    print(f"- 统计汇总（JSON）: {os.path.abspath(summary_path)}")


# ---------------------- 执行配置 ----------------------
# 输入文件：清洗后的纯代码文件
file_path1 = 'codegeex4-all-9b-sanitized.jsonl'
# 输出文件夹：存放结果的目录（自动创建）
output_dir = "mbpp_validation_results"  # 文件夹名称可自定义

# 执行完整流程
process_and_validate(file_path1, output_dir)