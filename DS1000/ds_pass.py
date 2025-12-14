import json
import re


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [json.loads(line.strip()) for line in file]
    return lines


def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')


def process_files(input_file, output_file):
    data = read_jsonl(input_file)
    for entry in data:
        if 'code' in entry:
            # 先删除 </code> 标签，再删除末尾的 END SOLUTION（不区分大小写、前后空格）
            code = entry['code'].replace('</code>', '')
            code = re.sub(r'\s*end\s+solution\s*$', '', code, flags=re.IGNORECASE)
            entry['code'] = code.strip()
    write_jsonl(data, output_file)
    print(f"清洗完成，文件保存至：{output_file}")


# 配置文件路径
input_file = 'codegeex4-all-9b_DS_1000.jsonl'
output_file = 'codegeex4-all-9b_DS_1000-sanitized.jsonl'

process_files(input_file, output_file)