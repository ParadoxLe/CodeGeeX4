import os
import json

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset

humaneval = load_dataset("openai_humaneval")

for index, data in enumerate(humaneval["test"]):
    data_id = data["task_id"]
    test_cases = data["test"]

    test_blocks = []
    current_block = []

    # 遍历每一行
    for line in test_cases.split('\n'):
        # 去掉该行的首尾空白（包括换行、空格、制表符）
        line_clean = line.strip()
        # 跳过空行（处理后无内容的行）
        if not line_clean:
            continue

        # 遇到assert行，处理上一个块
        if line_clean.startswith('assert'):
            if current_block:
                # 把当前块的所有内容拼接成一行（无任何换行）
                block_content = ' '.join(current_block)  # 用空格分隔，也可以直接''.join
                test_blocks.append(block_content)
                current_block = []
            # 加入当前assert行（已去掉首尾空白）
            current_block.append(line_clean)
        # 非assert行但属于当前块，加入列表
        elif current_block:
            current_block.append(line_clean)

    # 处理最后一个块
    if current_block:
        block_content = ' '.join(current_block)
        test_blocks.append(block_content)

    processed_item = {
        'task_id': data_id,
        'test': test_blocks,
    }

    with open("result/test.jsonl", 'a', encoding='utf-8') as f:
        json.dump(processed_item, f, ensure_ascii=False)
        f.write('\n')

print(f"结果已保存到:result/test.jsonl ")
