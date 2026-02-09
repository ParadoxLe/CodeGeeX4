import re
from typing import Any, Dict
import json
import yaml


def get_code(code_file: str) -> list[Any]:
    completion_list = []

    try:
        # 读取YAML文件（支持.yml/.yaml后缀）
        with open(code_file, "r", encoding="utf-8") as f:
            # 解析YAML文件，得到结构化数据（列表，每个元素是{task_id:..., completion:...}）
            yaml_data = yaml.safe_load(f)

        # 校验YAML数据格式
        if not isinstance(yaml_data, list):
            print(f"❌ YAML文件格式错误：根节点不是列表，而是{type(yaml_data)}")
            return completion_list

        # 遍历每个条目，提取completion
        for idx, item in enumerate(yaml_data):
            try:
                # 提取task_id和completion（兼容不同大小写/格式）
                task_id = item.get("task_id", f"未知任务_{idx + 1}")
                completion = item.get("completion", "")

                if completion.strip():  # 过滤空代码
                    completion_list.append(completion)
                    print(f"✅ 成功解析 {task_id}")
                else:
                    print(f"⚠️ {task_id} 的completion为空，跳过")

            except Exception as e:
                print(f"❌ 解析第{idx + 1}个条目失败：{str(e)}")
                print(f"   条目内容：{item}")

    except FileNotFoundError:
        print(f"❌ 文件不存在：{code_file}")
    except yaml.YAMLError as e:
        print(f"❌ YAML文件解析错误：{str(e)}")
    except Exception as e:
        print(f"❌ 读取文件时出错：{str(e)}")

    return completion_list


def get_test(test_file: str) -> list[Any]:
    data_list = []

    # 1. 读取整个文件并清理空白
    with open(test_file, "r", encoding="utf-8") as f:
        # 读取内容并去除所有换行/空格（只保留关键的}{分隔符）
        content = f.read().replace("\n", "").replace("  ", "").strip()

    # 2. 用}{分割成单个JSON对象字符串
    # 分割后每个片段需要补回对应的{}，比如分割后第一个片段是{...，最后一个是...}
    json_parts = content.split("}{")

    # 3. 修复每个片段的JSON格式
    for idx, part in enumerate(json_parts):
        if idx == 0:
            # 第一个片段：结尾补}
            fixed_json = part + "}"
        elif idx == len(json_parts) - 1:
            # 最后一个片段：开头补{
            fixed_json = "{" + part
        else:
            # 中间片段：开头补{，结尾补}
            fixed_json = "{" + part + "}"

        # 4. 解析修复后的JSON
        try:
            json_data = json.loads(fixed_json)
            task_id = json_data.get("task_id")
            test_cases = json_data.get("test", [])

            if task_id:
                data_list.append({
                    "task_id": task_id,
                    "test_cases": test_cases
                })
                print(f"✅ 成功提取 {task_id}（{len(test_cases)}条测试用例）")
        except json.JSONDecodeError as e:
            print(f"❌ 解析第{idx + 1}个对象失败：{fixed_json[:100]}... - {str(e)}")

    return data_list


def run_completion_test(code_list: list, test_list: list):
    # 初始化统计变量
    stats = {
        "total_code": len(code_list),  # 总代码数
        "all_passed": 0,  # 所有测试点都通过的代码数
        "assertion_failed": 0,  # 断言失败的代码数
        "code_error": 0,  # 代码本身出错的代码数
        "no_test_cases": 0,  # 无测试用例的代码数
        "function_not_found": 0  # 未找到函数的代码数
    }

    if not code_list or not test_list:
        print("没有获取到可执行的代码或测试用例")
        # 输出统计结果
        print_json(stats)
        return

    # 确保两个列表长度一致，取较小值
    max_idx = min(len(code_list), len(test_list))

    for idx in range(max_idx):
        print(f"\n{'=' * 50}")
        print(f"开始测试第 {idx + 1}/{max_idx} 个代码")
        print(f"{'=' * 50}")

        code_str = code_list[idx]
        test_item = test_list[idx]
        task_id = test_item.get("task_id", f"未知任务_{idx + 1}")

        # 标记当前代码的测试状态
        current_code_status = {
            "code_error": False,
            "assertion_failed": False,
            "all_passed": False,
            "no_test_cases": False,
            "function_not_found": False
        }

        try:
            # 清理多余引号、转义符，确保代码能正确执行
            cleaned_code = code_str.strip('"\'').strip('"""').strip("'''").strip()
            cleaned_code = cleaned_code.replace('\\n', '\n').replace('\\t', '\t')
            pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            func_name = re.findall(pattern, cleaned_code)

            if not func_name:
                print(f"❌ 未找到函数定义：{task_id}")
                current_code_status["function_not_found"] = True
                stats["function_not_found"] += 1
                continue

            func_name = func_name[0]

            # 核心修复：创建独立的临时命名空间，确保函数能被正确加载和获取
            temp_namespace = {}
            exec(cleaned_code, temp_namespace)  # 执行清理后的代码到临时命名空间
            print(f"✅ {task_id} 函数已加载！")

            # 准备测试用例
            test_cases = test_item.get("test_cases", [])
            # 处理空测试用例
            if not test_cases or test_cases is None:
                print(f"⚠️ {task_id} 未传入测试断言，跳过测试执行")
                current_code_status["no_test_cases"] = True
                stats["no_test_cases"] += 1
                continue

            processed_test_cases = []
            for case in test_cases:
                if isinstance(case, str):
                    processed_case = case.replace('candidate', func_name)
                    processed_test_cases.append(processed_case)
                else:
                    print(f"⚠️ {task_id} 测试用例格式错误（非字符串）：{case}，跳过")

            # 记录当前代码的断言失败数
            current_assert_fail = 0
            total_assert = len(processed_test_cases)

            for case_idx, assert_str in enumerate(processed_test_cases, 1):
                print(f"\n🔍 测试断言 {case_idx}/{total_assert}：{assert_str}")
                try:
                    # 在临时命名空间中执行断言语句
                    exec(assert_str, temp_namespace)
                    print(f"   ✅ 断言通过")
                except AssertionError:
                    print(f"   ❌ 断言失败：结果与预期不符")
                    current_assert_fail += 1
                except Exception as e:
                    print(f"   ❌ 断言执行出错：{e}")
                    print(f"      可用函数：{[k for k, v in temp_namespace.items() if callable(v)]}")
                    current_assert_fail += 1

            # 更新统计状态
            if current_assert_fail == 0 and total_assert > 0:
                print(f"\n🎉 {task_id} 所有测试点都通过了！")
                current_code_status["all_passed"] = True
                stats["all_passed"] += 1
            elif current_assert_fail > 0:
                print(f"\n❌ {task_id} 有 {current_assert_fail}/{total_assert} 个断言失败")
                current_code_status["assertion_failed"] = True
                stats["assertion_failed"] += 1

        except SyntaxError as e:
            print(f"❌ {task_id} 代码语法错误：{e}")
            print("📝 出错的代码内容（清理后）：")
            print(cleaned_code)
            current_code_status["code_error"] = True
            stats["code_error"] += 1
        except Exception as e:
            print(f"❌ {task_id} 代码运行出错：{e}")
            current_code_status["code_error"] = True
            stats["code_error"] += 1

    # 输出最终统计结果
    print("\n" + "=" * 60)
    print("测试结果统计报告")
    print("=" * 60)
    print_json(stats)


def print_json(stats: Dict):

    # 定义输出文件路径（可根据需要修改）
    output_file = "result/result.jsonl"

    # 构建结构化的统计数据字典
    statistics_data = {
        "整体统计": {
            "总代码数量": stats['total_code'],
            "所有测试点通过": stats['all_passed'],
            "所有测试点通过百分比": f"{stats['all_passed'] / stats['total_code'] * 100:.2f}%" if stats[
                                                                                                     'total_code'] > 0 else "0.00%",
            "断言失败": stats['assertion_failed'],
            "代码本身出错": stats['code_error'],
            "无测试用例": stats['no_test_cases'],
            "未找到函数定义": stats['function_not_found']
        },
        "有效测试统计": {}
    }

    # 计算有效测试率
    tested_code = stats['all_passed'] + stats['assertion_failed']
    denominator = stats['total_code'] - stats['code_error'] - stats['no_test_cases'] - stats['function_not_found']

    statistics_data["有效测试统计"] = {
        "完成测试的代码数": tested_code,
        "测试通过率": f"{tested_code / denominator * 100:.2f}%" if denominator > 0 else "0.00%"
    }

    try:
        # 关键修改：使用覆盖模式（w）打开文件，每次运行都会清空原有内容
        with open(output_file, "w", encoding="utf-8") as f:
            # 将统计数据转换为 JSON 字符串并写入（符合 JSONL 格式）
            json_line = json.dumps(statistics_data, ensure_ascii=False, indent=2)
            f.write(json_line + "\n")  # 每行一个 JSON 对象，末尾加换行符

        # 控制台提示写入成功
        print(f"\n✅ 统计结果已覆盖写入文件：{output_file}")
        print(f"   本次统计核心数据：总代码数 {stats['total_code']}，全通过 {stats['all_passed']} 个")

    except Exception as e:
        # 捕获文件写入异常并提示
        print(f"❌ 写入统计文件失败：{str(e)}")


# ------------------- 主程序 -------------------
if __name__ == "__main__":
    code_path = "result/generate.yaml"
    test_path = "result/test.jsonl"

    # 1. 读取 completion 代码
    code_content = get_code(code_path)
    # print(code_content)
    # 2. 读取 test 代码
    test_content = get_test(test_path)
    # print(test_content)
    # 3.传入两个数组进行测试
    run_completion_test(code_content, test_content)
