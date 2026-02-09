import os
import json

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset

mbpp = load_dataset("google-research-datasets/mbpp")


# 遍历MBPP的test子集（MBPP的测试集是"test"分卷）
for index, data in enumerate(mbpp["test"]):
    data_id = data["task_id"]  # MBPP同样有task_id字段
    test_cases_list = data["test_list"]  # MBPP的测试用例存储在test_list中（列表类型）

    test_blocks = []
    # 遍历test_list中的每一个测试用例字符串
    for test_case in test_cases_list:
        current_block = []
        # 按行拆分单个测试用例
        for line in test_case.split('\n'):
            line_clean = line.strip()
            if not line_clean:
                continue

            # 遇到assert行，处理上一个块（逻辑和原代码一致）
            if line_clean.startswith('assert'):
                if current_block:
                    block_content = ' '.join(current_block)
                    test_blocks.append(block_content)
                    current_block = []
                current_block.append(line_clean)
            elif current_block:
                current_block.append(line_clean)

        # 处理每个测试用例的最后一个块
        if current_block:
            block_content = ' '.join(current_block)
            test_blocks.append(block_content)

    # 构造和原代码格式一致的输出项
    processed_item = {
        'task_id': data_id,
        'test': test_blocks,
    }

    # 追加写入JSONL文件（和原代码路径一致）
    with open("result/test.jsonl", 'a', encoding='utf-8') as f:
        json.dump(processed_item, f, ensure_ascii=False)
        f.write('\n')

print(f"结果已保存到: result/test.jsonl")