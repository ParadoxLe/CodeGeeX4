import json
import os
import warnings
from datetime import datetime
from typing import Tuple

# 屏蔽HuggingFace的FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# 设置镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset


# 基础读写函数
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line.strip()) for line in file]


def write_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')


def write_summary_json(summary_dict, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=4)
    print(f"\n统计汇总已保存到：{os.path.abspath(file_path)}")


# 核心验证函数（适配DS-1000字符串型code_context）
def run_tests(code: str, code_context_str: str) -> Tuple[bool, str]:
    exec_env = {}
    try:
        # 执行code_context加载测试函数和环境
        exec(code_context_str, exec_env)

        # 提取exec_context（插入生成代码的模板）
        if "exec_context" in exec_env:
            exec_context = exec_env["exec_context"]
        else:
            import re
            exec_context_match = re.search(r'exec_context = r"""(.*?)"""', code_context_str, re.DOTALL)
            if not exec_context_match:
                return False, "无法从code_context提取执行模板"
            exec_context = exec_context_match.group(1)

        # 插入生成的代码并执行
        full_code = exec_context.replace("[insert]", code)
        exec(full_code, exec_env)
    except SyntaxError as e:
        return False, f"语法错误: 行{e.lineno} - {str(e)[:100]}"
    except Exception as e:
        return False, f"代码执行错误: {type(e).__name__} - {str(e)[:100]}"

    # 执行测试用例断言
    try:
        if "generate_test_case" not in exec_env or "exec_test" not in exec_env:
            return False, "code_context缺少测试必需函数"

        test_input, expected_result = exec_env["generate_test_case"](1)
        test_pass = exec_env["exec_test"](exec_env.get("result"), expected_result)
        if not test_pass:
            return False, "测试用例失败: 结果与预期不一致"
    except Exception as e:
        return False, f"测试用例运行错误: {type(e).__name__} - {str(e)[:100]}"

    return True, "所有测试用例通过"


# 主函数（完全适配你的数据格式）
def process_and_validate(file_path1, output_dir="ds1000_validation_results", ds1000_dataset_path="xlangai/DS-1000"):
    # 创建结果文件夹
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存到文件夹：{os.path.abspath(output_dir)}")

    # 读取数据（你的清洗后文件 + DS-1000数据集）
    generated_data = read_jsonl(file_path1)
    ds1000 = load_dataset(ds1000_dataset_path)
    test_dataset = ds1000["test"]

    # 提取数据集信息（ID：metadata.problem_id）
    dataset_problem_ids = [item["metadata"]["problem_id"] for item in test_dataset]
    dataset_total = len(test_dataset)
    dataset_id_min = min(dataset_problem_ids) if dataset_problem_ids else None
    dataset_id_max = max(dataset_problem_ids) if dataset_problem_ids else None
    unique_libraries = list(set([item["metadata"]["library"] for item in test_dataset]))

    print(f"\n=== DS-1000 数据集信息 ===")
    print(f"test分割总题目数：{dataset_total}")
    print(f"匹配ID字段：metadata.problem_id")
    print(f"ID范围：{dataset_id_min} - {dataset_id_max}")
    print(f"包含语言/库：{unique_libraries}")
    print(f"前5个ID示例：{dataset_problem_ids[:5]}")
    print(f"========================\n")

    # 构建映射：problem_id → 测试信息（适配数据集结构）
    id_to_test_info = {
        item["metadata"]["problem_id"]: {
            "code_context_str": item["code_context"],
            "library": item["metadata"]["library"]
        } for item in test_dataset
    }

    # 初始化统计变量
    updated_data = []
    total_passed = 0
    total = len(generated_data)
    unknown_id_count = 0
    error_stats = {
        "缺少problem_id字段": 0,
        "problem_id格式错误": 0,
        "未知题目ID": 0,
        "无有效代码": 0,
        "语法错误": 0,
        "代码执行错误": 0,
        "测试用例失败": 0,
        "测试环境错误": 0
    }
    library_stats = {lib: {"总题数": 0, "通过数": 0} for lib in unique_libraries}

    # 逐题验证（完全匹配你的数据字段）
    for entry in generated_data:
        new_entry = entry.copy()

        # 1. 读取你数据中真正匹配数据集的ID字段：problem_id
        original_problem_id = entry.get("metadata", {}).get("problem_id")
        if original_problem_id is None:
            new_entry['validation_result'] = "生成数据缺少problem_id字段（核心匹配ID）"
            new_entry['passed'] = False
            error_stats["缺少problem_id字段"] += 1
            updated_data.append(new_entry)
            print(f"处理题目（无problem_id）→ 失败: {new_entry['validation_result']}")
            continue

        # 2. 处理problem_id（确保是整数，匹配数据集）
        try:
            problem_id = int(original_problem_id)
        except ValueError:
            new_entry['validation_result'] = f"problem_id格式错误（需为整数）：{original_problem_id}"
            new_entry['passed'] = False
            error_stats["problem_id格式错误"] += 1
            updated_data.append(new_entry)
            print(f"处理题目 problem_id={original_problem_id} → 失败: {new_entry['validation_result']}")
            continue

        # 3. 读取你数据中的代码字段：code（已清洗）
        cleaned_code = entry.get('code', '').strip()
        new_entry['cleaned_code'] = cleaned_code  # 保留清洗后代码

        # 4. 验证逻辑（匹配数据集）
        if problem_id not in id_to_test_info:
            new_entry['validation_result'] = f"未知题目ID（数据集无该metadata.problem_id）：{problem_id}"
            new_entry['passed'] = False
            unknown_id_count += 1
            error_stats["未知题目ID"] += 1
        else:
            test_info = id_to_test_info[problem_id]
            code_context_str = test_info["code_context_str"]
            library = test_info["library"]

            # 更新按库统计
            if library in library_stats:
                library_stats[library]["总题数"] += 1

            # 检查代码是否有效
            if not cleaned_code:
                new_entry['validation_result'] = "无有效代码（清洗后为空）"
                new_entry['passed'] = False
                error_stats["无有效代码"] += 1
            else:
                # 执行验证
                passed, msg = run_tests(cleaned_code, code_context_str)
                new_entry['validation_result'] = msg
                new_entry['passed'] = passed
                new_entry['library'] = library  # 记录所属库
                if passed:
                    total_passed += 1
                    if library in library_stats:
                        library_stats[library]["通过数"] += 1
                else:
                    # 统计具体错误类型
                    if "语法错误" in msg:
                        error_stats["语法错误"] += 1
                    elif "代码执行错误" in msg:
                        error_stats["代码执行错误"] += 1
                    elif "测试用例失败" in msg:
                        error_stats["测试用例失败"] += 1
                    else:
                        error_stats["测试环境错误"] += 1

        updated_data.append(new_entry)
        # 实时打印进度（显示核心ID：problem_id）
        print(
            f"处理题目 problem_id={original_problem_id} → {'通过' if new_entry['passed'] else '失败'}: {new_entry['validation_result'][:100]}"
        )

    # 计算核心统计信息
    pass_rate = (total_passed / total) * 100 if total > 0 else 0.0
    failed_count = total - total_passed

    # 构造统计汇总
    summary_dict = {
        "验证状态": "完成",
        "验证时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "核心统计": {
            "总题目数": total,
            "通过题目数": total_passed,
            "失败题目数": failed_count,
            "未知ID题目数": unknown_id_count,
            "Pass@1 通过率(%)": round(pass_rate, 2)
        },
        "错误类型统计": error_stats,
        "按库分类统计": {
            lib: {
                "总题数": stats["总题数"],
                "通过数": stats["通过数"],
                "通过率(%)": round((stats["通过数"] / stats["总题数"]) * 100, 2) if stats["总题数"] > 0 else 0
            } for lib, stats in library_stats.items()
        },
        "数据集信息": {
            "数据集名称": ds1000_dataset_path,
            "分割名称": "test",
            "数据集总题目数": dataset_total,
            "匹配ID字段路径": "metadata.problem_id",
            "code_context类型": "字符串（含测试函数和执行环境）"
        },
        "生成数据信息": {
            "核心匹配ID字段": "problem_id",  # 你的数据中真正用于匹配的字段
            "代码字段": "code",
            "其他字段": "id、library_problem_id、library等"
        },
        "文件信息": {
            "输入文件路径": os.path.abspath(file_path1),
            "完整结果文件路径": os.path.join(output_dir, "codegeex4-all-9b-DS1000-validated-full.jsonl"),
            "统计汇总文件路径": os.path.join(output_dir, "ds1000_validation_summary.json")
        }
    }

    # 保存结果文件
    full_result_path = os.path.join(output_dir, "codegeex4-all-9b-DS1000-validated-full.jsonl")
    summary_path = os.path.join(output_dir, "ds1000_validation_summary.json")
    write_jsonl(updated_data, full_result_path)
    write_summary_json(summary_dict, summary_path)

    # 终端打印最终汇总
    print(f"\n=== 验证完成 ===")
    print(f"总题目数: {total}")
    print(f"通过题目数: {total_passed}")
    print(f"失败题目数: {failed_count}")
    print(f"Pass@1 通过率: {pass_rate:.2f}%")
    print(f"\n按库分类统计：")
    for lib, stats in library_stats.items():
        if stats["总题数"] > 0:
            print(f"- {lib}: 总题数{stats['总题数']}，通过{stats['通过数']}，通过率{stats['通过率(%)']:.2f}%")
        else:
            print(f"- {lib}: 总题数{stats['总题数']}，通过{stats['通过数']}")
    print(f"\n错误类型统计：")
    for err_type, count in error_stats.items():
        print(f"- {err_type}: {count}个")
    print(f"\n文件保存位置：")
    print(f"- 完整结果（含所有题目验证信息）: {os.path.abspath(full_result_path)}")
    print(f"- 统计汇总（JSON格式）: {os.path.abspath(summary_path)}")


# ---------------------- 执行配置（无需修改，直接运行） ----------------------
# 输入文件：你的清洗后数据文件
file_path1 = 'codegeex4-all-9b_DS_1000-sanitized.jsonl'
# 输出文件夹：自动创建，存放验证结果
output_dir = "ds1000_validation_results"

# 执行验证流程
if __name__ == "__main__":
    process_and_validate(file_path1, output_dir)