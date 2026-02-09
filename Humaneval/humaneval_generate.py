import os
import torch
import yaml

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer  # 直接导入transformers库

dataset_path = "openai_humaneval"
# 加载CodeGeeX4模型和分词器
model_path = "zai-org/codegeex4-all-9b"  # CodeGeeX4模型路径
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to(device).eval()

humaneval = load_dataset(dataset_path)
print("模型为" + model_path)
print("数据集为" + dataset_path)

# 存储最后生成的所有代码
all_results = []


def clean_code(raw_code: str) -> str:
    # 移除开头的```python标记和末尾的```
    code = raw_code.replace("```python", "").replace("```", "").strip()

    # 处理所有三重双引号包裹的内容（支持多对"""）
    parts = code.split('"""')
    clean_parts = []
    # 遍历分割后的部分：保留奇数索引（0、2、4...）的内容，跳过偶数索引（1、3、5...）的注释
    for idx, part in enumerate(parts):
        if idx % 2 == 0:  # 只保留三重引号外的部分（索引0、2、4...）
            clean_parts.append(part)
    cleaned_code = ''.join(clean_parts).strip()

    # 移除多余的空行，保持代码整洁
    lines = [line for line in cleaned_code.split('\n') if line.strip()]
    return '\n'.join(lines)


for index, data in enumerate(humaneval["test"]):
    print(f"处理序号为:{index}")
    print("当前问题为:" + data["prompt"])
    question = data["prompt"]
    data_id = data["task_id"]
    test_cases = data["test"]

    content = f"""你是一位智能编程助手，你叫CodeGeeX。你会为用户实现以下Python函数，只提供格式规范、可以执行、准确安全的实现代码，并参考问题中给出的测试案例，不输出，{question}"""

    message = [
        {'role': 'user', 'content': content}
    ]
    inputs = tokenizer.apply_chat_template(message,
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )
        answer = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        answer = clean_code(answer)

        print(f"Answer:\n{answer}\n")
        # 收集结果
        result_item = {"task_id": f"{data_id}", "completion": f"{answer} "}
        all_results.append(result_item)

with open('result/generate.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(all_results, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

print("All data has been saved to result/generate.yaml")
