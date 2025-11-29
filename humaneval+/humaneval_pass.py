import json
import re


def read_jsonl(file_path):
    """读取JSONL文件（和原脚本完全一致）"""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [json.loads(line.strip()) for line in file]
    return lines


def write_jsonl(data, file_path):
    """写入JSONL文件（和原脚本完全一致）"""
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')


def extract_code_blocks(solution):
    # 核心函数：只提取第二个 """ 后面的所有内容
    pattern = r'"""(.*?)"""(.*)```'
    match = re.search(pattern, solution, re.DOTALL)  # re.DOTALL让.匹配换行
    if match and match.group(2):
        # 返回第二个 """ 后的内容，保留原始格式（仅去除首尾空行/空格）
        return match.group(2).strip()
    return ""  # 未找到两个 """ 则返回空


# 主函数：清洗HumanEval生成结果中的代码
def process_files(file_path1, output_file):
    data1 = read_jsonl(file_path1)
    updated_data = []

    for entry in data1:
        new_entry = entry.copy()  # 避免修改原字典
        # 提取第二个 """ 后的内容，更新solution字段
        cleaned_content = extract_code_blocks(entry['solution'])
        new_entry['solution'] = cleaned_content
        updated_data.append(new_entry)

    write_jsonl(updated_data, output_file)


# ---------------------- 配置文件路径 ----------------------
# 输入文件：你生成的HumanEval原始结果（codegeex4-all-9b_HumanEval.jsonl）
file_path1 = 'codegeex4-all-9b_HumanEval.jsonl'
# 输出文件：清洗后的纯代码文件（命名保持原风格）
output_file = 'codegeex4-all-9b_HumanEval-sanitized.jsonl'

# 执行清洗
process_files(file_path1, output_file)

print(f"代码清洗完成！")
print(f"输入文件：{file_path1}")
print(f"输出文件：{output_file}")
