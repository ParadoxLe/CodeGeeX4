import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset

# 加载DS-1000数据集
ds = load_dataset("xlangai/DS-1000")
test_dataset = ds["test"]

# 打印第一个样本的所有字段名（关键！）
print("DS-1000 测试集第一个样本的字段名：")
print(list(test_dataset[0].keys()))
print("\n第一个样本的完整内容：")
print(test_dataset[0])