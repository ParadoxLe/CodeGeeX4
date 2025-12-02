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


def format_code(code):
    """新增：格式化代码，兼容「有from」「只有def」两种场景"""
    if not code:
        return ""

    # 步骤1：区分场景处理 import/from 语句（只有存在时才处理）
    if code.startswith(('from ', 'import ')):
        # 在 import/from 语句后添加换行（避免和def同行）
        code = re.sub(r'(from .*? import .*?)(def .*)', r'\1\n\2', code)
        code = re.sub(r'(import .*?)(def .*)', r'\1\n\2', code)

    # 步骤2：在 def 函数头后添加换行和4空格缩进（通用）
    code = re.sub(r'(def .*?:)', r'\1\n    ', code)

    # 步骤3：在 for/if/while 等语句后添加换行和8空格缩进（通用）
    code = re.sub(r'(for .*?:|if .*?:|while .*?:)', r'\1\n        ', code)

    # 步骤4：在 return 前添加换行和8空格缩进（避免和循环/条件同行）
    code = re.sub(r'(?<!\n)(?<!        )(return .*)', r'\n        \1', code)

    # 步骤5：合并多余空行，清理首尾空格（通用）
    code = re.sub(r'\n+', '\n', code).strip()

    return code


def extract_code_blocks(solution):
    # 按需求清洗：
    # 1. 删除两个 """中间的内容（包括前后的"""）；
    # 2. 兼容两种场景：有from则保留from及之后，只有def则保留def及之后；
    # 3. 精准删除末尾的 ``` 标记；
    # 4. 格式化代码；

    # 步骤1：删除两个 """ 中间的内容（包括 """ 本身）
    pattern_remove_triple_quote = r'"""(.*?)"""'
    code = re.sub(pattern_remove_triple_quote, '', solution, flags=re.DOTALL)

    # 步骤2：兼容「有from」「只有def」两种场景，删除前面的无关内容
    # 正则逻辑：优先匹配 from（有则保留from及之后），没有则匹配def（保留def及之后）
    pattern_remove_before = r'^.*?(?=from|def)'  # 关键：用 | 同时匹配 from 或 def
    code = re.sub(pattern_remove_before, '', code, flags=re.DOTALL)

    # 步骤3：精准删除末尾的 ``` 标记（兼容两种开头）
    pattern_remove_trailing_backticks = r'^\s*(from.*?|def.*?)\s*```\s*$'  # 同时匹配 from 或 def 开头
    match = re.match(pattern_remove_trailing_backticks, code, flags=re.DOTALL)
    if match:
        code = match.group(1)  # 只保留核心代码，去掉末尾 ```
    else:
        code = code.strip()

    # 步骤4：格式化代码（兼容两种场景）
    return format_code(code)


# 主函数：清洗HumanEval生成结果中的代码
def process_files(file_path1, output_file):
    data1 = read_jsonl(file_path1)
    updated_data = []

    for entry in data1:
        new_entry = entry.copy()  # 避免修改原字典
        cleaned_content = extract_code_blocks(entry['solution'])
        new_entry['solution'] = cleaned_content
        updated_data.append(new_entry)

    write_jsonl(updated_data, output_file)


# ---------------------- 配置文件路径 ----------------------
file_path1 = 'codegeex4-all-9b_HumanEval.jsonl'
output_file = 'codegeex4-all-9b_HumanEval-sanitized.jsonl'

# 执行清洗
process_files(file_path1, output_file)

print(f"代码清洗完成！")
print(f"输入文件：{file_path1}")
print(f"输出文件：{output_file}")