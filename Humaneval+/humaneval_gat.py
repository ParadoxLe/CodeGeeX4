import os
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, ".."))
sys.path.append(project_root)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import warnings
from EnhanceModel.gat_module.ModelWithGAT import CodeGeeX4WithGAT

# 屏蔽HuggingFace的FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# 设置镜像源（需在加载模型前执行）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
from datasets import load_dataset
from transformers import AutoTokenizer  # 直接导入transformers库

model_path = "zai-org/codegeex4-all-9b"  # CodeGeeX4模型路径
print(f"直接加载模型：{model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = CodeGeeX4WithGAT.from_pretrained(model_path,
                                         gat_num_heads=8,
                                         gat_dropout=0.1,
                                         gat_enabled=True,  # 可以设置为False来禁用GAT
                                         trust_remote_code=True)
model.eval()  # 切换到推理模式
HumanEval = load_dataset("evalplus/humanevalplus")

answers = []

for (index, data) in enumerate(HumanEval["test"]):
    print(f"Working on {index}\n")
    print(f"Original question:\n{data['prompt']}\n")
    question = data['prompt'].strip()
    content = f"""Write a solution to the following problem:
        ```python
        {question}
        ```"""
    messages = [
        {'role': 'user', 'content': content}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt").to(model.device)

    outputs = model.generate(inputs,
                             max_new_tokens=1024,
                             do_sample=True,
                             temperature=0.2,
                             top_p=0.95,
                             num_return_sequences=1,
                             eos_token_id=tokenizer.eos_token_id)

    answer = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    print(f"Answer:\n{answer}\n")
    json_data = {"task_id": f"HumanEval/{index}",
                 "solution": f"{answer}"}

    # Save results to a JSON file
    with open('codegeex4-all-9b_HumanEval.jsonl', 'a') as f:
        json.dump(json_data, f)
        f.write('\n')

print("All data has been saved to codegeex4-all-9b_HumanEval.jsonl")
