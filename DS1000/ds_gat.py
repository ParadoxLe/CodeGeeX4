import os
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, ".."))
sys.path.append(project_root)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from EnhanceModel.gat_module.ModelWithGAT import CodeGeeX4WithGAT

model_path = "zai-org/codegeex4-all-9b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = CodeGeeX4WithGAT.from_pretrained(model_path,
                                         gat_num_heads=8,
                                         gat_dropout=0.1,
                                         gat_enabled=True,  # 可以设置为False来禁用GAT
                                         trust_remote_code=True)
model.eval()  # 切换到推理模式
model_identifier = os.path.basename(model_path)

ds_1000 = load_dataset("xlangai/DS-1000")

answers = []

for (index, data) in enumerate(ds_1000["test"]):
    print(f"Working on {index}\n")
    question = data['prompt'].strip()
    metadata = data['metadata']
    data_id = index
    if "SOLUTION START" in question:
        question = question.strip()
    else:
        question = question.strip()
    print(f"Input question:\n{question}\n")
    content = f"""Please Help me to finish the following code completion and Place the executable code between <code> 
                and </code> tags, without any other non-executable things.
            {question}
            """
    messages = [
        {'role': 'user', 'content': content}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs,
                             max_new_tokens=1024,
                             do_sample=False,
                             temperature=0.0,
                             top_p=1.0,
                             num_return_sequences=1,
                             eos_token_id=tokenizer.eos_token_id)

    answer = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    print(f"Answer:\n{answer}\n")
    json_data = {"id": data_id,
                 "code": answer,
                 "metadata": metadata}

    with open(f'{model_identifier}_DS_1000.jsonl', 'a') as f:
        json.dump(json_data, f)
        f.write('\n')

print(f"All data has been saved to /{model_identifier}_DS_1000.jsonl")
print(f"All data has been saved to /{model_identifier}_DS_1000.jsonl")
