import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset

# 加载HumanEval+数据集
humaneval = load_dataset("evalplus/humanevalplus")
test_dataset = humaneval["test"]

# 取第一个样本，打印所有字段名
first_sample = test_dataset[0]
print("HumanEval+ 数据集字段名：")
for field in first_sample.keys():
    print(f"- {field}")