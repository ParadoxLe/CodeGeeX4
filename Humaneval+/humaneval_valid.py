import os
import sys
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, ".."))
sys.path.append(project_root)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import warnings
# 屏蔽HuggingFace的FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# 设置镜像源（需在加载模型前执行）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
from datasets import load_dataset
from Model_Enhancer.model_loader import load_code_model,load_code_model_GNN,load_code_model_Reflect

enhanceModel = load_code_model()
tokenizer = enhanceModel.tokenizer
model = enhanceModel.model
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
