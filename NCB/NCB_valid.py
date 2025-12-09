import sys
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, ".."))
sys.path.append(project_root)
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # 直接导入transformers库

# 直接加载CodeGeeX4模型和分词器
model_path = "zai-org/codegeex4-all-9b"  # CodeGeeX4模型路径
print(f"直接加载模型：{model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto"  # 自动分配设备（CPU/GPU）
)
model.eval()  # 切换到推理模式

# 配置要处理的语言和自然语言类型
LANGUAGES = ["java"]  # 要评估的编程语言
NATURAL_LANGS = ["zh", "en"]  # 要评估的自然语言（中文/英文）
MODEL_NAME = model_path.split("/")[-1]

# 创建结果根目录
results_root = Path(f"NaturalCodeBench/results/{MODEL_NAME}")
results_root.mkdir(parents=True, exist_ok=True)

# 遍历所有语言组合处理
for lang in LANGUAGES:
    for nat_lang in NATURAL_LANGS:
        # 问题数据路径
        problem_file = f"NaturalCodeBench/problems/ncb_{lang}_{nat_lang}.jsonl"
        if not os.path.exists(problem_file):
            print(f"警告：未找到问题文件 {problem_file}，跳过该场景\n")
            continue

        # 加载问题数据（逐行读取JSONL）
        print(f"加载问题数据：{problem_file}")
        problems = [json.loads(line) for line in open(problem_file, "r", encoding="utf-8")]

        # 定义输出文件路径
        output_file = results_root / f"{MODEL_NAME}_ncb_{lang}_{nat_lang}.jsonl"

        # 逐行处理并追加写入结果
        for (index, data) in enumerate(problems):
            print(f"Working on {index + 1}/{len(problems)} ({lang}-{nat_lang})\n")
            print(f"Original prompt:\n{data['prompt']}\n")

            # 提取核心数据
            data_id = data["_id"]
            prompt = data["prompt"].strip()

            # 构造对话prompt（适配NCB数据集）
            content = f"""请仅生成可执行的代码，不要添加任何解释性文字、注释或多余内容。
需求：{prompt}
要求：代码语法正确，能通过所有测试用例，可直接运行。"""

            messages = [
                {'role': 'user', 'content': content}
            ]

            # 构造输入
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                chat_format="chatglm",  # 显式指定格式，适配CodeGeex4
                return_tensors="pt"
            ).to(model.device)

            # 生成代码
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.15,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    num_beams=4,  # 推荐值：3~4（不宜过高，否则速度大幅下降）
                    early_stopping=True,  # 当波束搜索找到最优候选时提前停止
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )

            # 解码并清洗生成结果
            answer = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).strip()
            # 清洗markdown代码块
            if answer.startswith("```"):
                answer = answer.split("```")[1].strip()
                if answer.startswith(("python", "java")):
                    answer = answer.split("\n", 1)[1].strip()

            print(f"Generated code:\n{answer}\n")

            # 构造结果数据
            json_data = {
                "_id": data_id,
                "response": answer
            }

            # 追加写入JSONL文件
            with open(output_file, 'a', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False)
                f.write('\n')

        print(f"该场景结果已保存至：{output_file}\n")

# 输出评估命令提示
print(f"All data has been saved to {results_root}")
print("可运行评估命令：")
print(
    f"python ncb/evaluate.py --languages {' '.join(LANGUAGES)} --natural_langs {' '.join(NATURAL_LANGS)} --ckpt_name {MODEL_NAME} --num_workers 16 --ks 1")
