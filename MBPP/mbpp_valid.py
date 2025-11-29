import json
import os
import warnings
# 屏蔽HuggingFace的FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# 设置镜像源（需在加载模型前执行）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "zai-org/codegeex4-all-9b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             device_map="auto", trust_remote_code=True)

mbpp = load_dataset("evalplus/mbppplus")

answers = []

for (index, data) in enumerate(mbpp["test"]):
    print(f"Working on {index}\n")
    print(f"Original question:\n{data['prompt']}\n")
    question = data['prompt'].strip()
    data_id = data['task_id']
    assertion = data['test_list']
    content = f"""{question}
                Your code should satisfy the following assertion:
                ```python
                {assertion}
                ```
                """
    messages = [
        {'role': 'user', 'content': content}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs,
                             max_new_tokens=1024,
                             do_sample=False,
                             temperature=0.2,
                             top_p=0.95,
                             num_return_sequences=1,
                             eos_token_id=tokenizer.eos_token_id)

    answer = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    print(f"Answer:\n{answer}\n")
    json_data = {"task_id": f"Mbpp/{data_id}",
                 "solution": f"{answer}"}

    # Save results to a JSON file
    with open('codegeex4-all-9b.jsonl', 'a') as f:
        json.dump(json_data, f)
        f.write('\n')

print("All data has been saved to codegeex4-all-9b.jsonl")
