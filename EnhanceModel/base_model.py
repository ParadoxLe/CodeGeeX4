import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer  # 直接导入transformers库

# 直接加载CodeGeeX4模型和分词器
model_path = "zai-org/codegeex4-all-9b"  # CodeGeeX4模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto"  # 自动分配设备（CPU/GPU）
)

print(model) #查看模型整体结构