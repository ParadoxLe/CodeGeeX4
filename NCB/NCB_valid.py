import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, ".."))
sys.path.append(project_root)
import json
from pathlib import Path
import torch

from Model_Enhancer.model_loader import load_code_model, load_code_model_GNN, load_code_model_Reflect,load_code_model_GNN_Reflect


def load_model_and_tokenizer(model_path):
    """加载Hugging Face格式的模型和分词器"""
    enhanceModel = load_code_model_GNN_Reflect()
    tokenizer = enhanceModel.tokenizer
    model = enhanceModel.model
    model.eval()  # 推理模式
    return tokenizer, model


def generate_code(tokenizer, model, prompt):
    """
    根据提示词生成代码（修复对话格式错误 + 适配NCB数据集）
    """
    # 将纯文本prompt封装为标准对话列表
    messages = [
        {
            "role": "user",
            "content": f"""请仅生成可执行的代码，不要添加任何解释性文字、注释或多余内容。
需求：{prompt}
要求：代码语法正确，能通过所有测试用例，可直接运行。"""  # 优化NCB生成指令
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,  # 传入对话列表（替代原纯文本prompt）
        add_generation_prompt=True,
        chat_format="chatglm",  # 显式指定格式，适配CodeGeex4
        return_tensors="pt"
    ).to(model.device)

    # 生成参数（可根据模型特点调整）
    with torch.no_grad():
        # 修复：移除**inputs（inputs是tensor，无需解包）
        outputs = model.generate(
            inputs,
            max_new_tokens=1024,
            temperature=0.2,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 提取生成的代码（去除输入prompt部分）
    generated_text = tokenizer.decode(
        outputs[0][len(inputs[0]):],
        skip_special_tokens=True
    ).strip()

    # ========== 适配NCB：清洗生成结果（去除markdown代码块） ==========
    if generated_text.startswith("```"):
        generated_text = generated_text.split("```")[1].strip()
        # 去除python/java语言标记
        if generated_text.startswith(("python", "java")):
            generated_text = generated_text.split("\n", 1)[1].strip()

    return generated_text


def process_problems(model_name, tokenizer, model, languages=["python", "java"], natural_langs=["zh", "en"]):
    """
    处理所有问题并生成结果文件
    """
    # 创建结果目录（按模型名称区分）
    results_root = Path(f"NaturalCodeBench/results/{model_name}")
    results_root.mkdir(parents=True, exist_ok=True)

    for lang in languages:
        for nat_lang in natural_langs:
            # 问题数据路径（对应problems或data目录下的文件）
            problem_file = f"NaturalCodeBench/problems/ncb_{lang}_{nat_lang}.jsonl"  # 若问题在data目录则改为data/{lang}_{nat_lang}/ncb_{lang}_{nat_lang}.jsonl
            if not os.path.exists(problem_file):
                print(f"警告：未找到问题文件 {problem_file}，跳过该场景")
                continue

            # 加载问题数据
            print(f"加载问题数据：{problem_file}")
            problems = [json.loads(line) for line in open(problem_file, "r", encoding="utf-8")]

            # 生成代码结果
            results = []
            for idx, problem in enumerate(problems):
                _id = problem["_id"]
                prompt = problem["prompt"]  # 问题提示词

                print(f"生成第 {idx + 1}/{len(problems)} 个问题（ID: {_id}，{lang}-{nat_lang}）")
                try:
                    generated_code = generate_code(tokenizer, model, prompt)
                except Exception as e:
                    print(f"生成失败（ID: {_id}）：{str(e)}")
                    generated_code = ""

                # 按要求格式保存（仅包含_id和response）
                results.append({
                    "_id": _id,
                    "response": generated_code
                })

            # 保存结果文件
            output_file = results_root / f"{model_name}_ncb_{lang}_{nat_lang}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"结果已保存至：{output_file}\n")


if __name__ == "__main__":
    # 配置参数
    MODEL_NAME = "zai-org/codegeex4-all-9b"
    LANGUAGES = ["java"]  # 要评估的编程语言
    NATURAL_LANGS = ["zh", "en"]  # 要评估的自然语言（中文/英文）

    # 加载模型和分词器
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME)

    # 生成并保存结果
    process_problems(MODEL_NAME.split("/")[-1], tokenizer, model, LANGUAGES, NATURAL_LANGS)

    print("所有场景生成完成！可运行评估命令：")
    print(
        f"python ncb/evaluate.py --languages {' '.join(LANGUAGES)} --natural_langs {' '.join(NATURAL_LANGS)} --ckpt_name {MODEL_NAME.split('/')[-1]} --num_workers 16 --ks 1 10 100")

# python ncb/evaluate.py --languages java --natural_langs zh en --ckpt_name codegeex4-all-9b --num_workers 16 --ks 1
