"""
Run a trained model to generate Python code with CodeGeeX4.
"""

import io
import json
import logging
import math
import random
import numpy as np
import os
import pprint
import sys
import time
import transformers
import torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from reindent import run as run_reindent
# 移除GPT2相关导入，保留通用导入
from transformers import AutoTokenizer, AutoModelForCausalLM

# for timing and debugging
from datetime import datetime, date
from tqdm import tqdm


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr,
        ret,
        config={
            "dry-run": False,
            "help": False,
            "to": 10,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 10,
            "all-tabs": False
        }
    )

    return ret.getvalue()


def generate_prompt(args, test_case, prompt, solutions, tokenizer, starter_code=None):
    _input = "\nQUESTION:\n"
    data = prompt
    _input += data
    if starter_code is not None:
        data = starter_code
        data = "\n" + data
        _input += data

    data = test_case
    if not data.get("fn_name"):
        _input += "\nUse Standard Input format"
    else:
        _input += "\nUse Call-Based format"

    _input += "\nANSWER:\n"

    if args.peeking > 0.0:
        # 修复变量名错误：peek_frac -> peeking
        sols = solutions
        sample_sol = random.choice(sols)
        rand_sol = reindent_code(sample_sol)
        rand_sol = tokenizer.encode(rand_sol, verbose=False)
        tokens_taken = int(args.peeking * len(rand_sol))  # 修复变量名
        rand_sol = rand_sol[:tokens_taken]
        _input += tokenizer.decode(rand_sol)
    else:
        sample_sol = None

    return _input, sample_sol


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # 加载数据集
    problems = load_dataset("codeparrot/apps", split=f"{args.split}")

    gpt_codes = {}
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    if not args.end:
        codes_loc = os.path.join(args.save, f"all_codes.json")
    else:
        codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes.json")

    # 筛选指定范围的问题
    if args.index:
        problems = load_dataset("codeparrot/apps", split=f"{args.split}[{args.index}]")
    else:
        if args.start > len(problems) or args.start < 0:
            print(f"start index {args.start} > number of problems {len(problems)}")
            return
        start = args.start
        end = args.end if args.end and args.end <= len(problems) else len(problems)
        problems = load_dataset("codeparrot/apps", split=f"{args.split}[{start}:{end}]")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载CodeGeeX4模型和分词器
    print("Loading CodeGeeX4 model and tokenizer...")
    if args.load:
        # 加载本地的CodeGeeX4模型
        tokenizer = AutoTokenizer.from_pretrained(args.load, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.load,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
    else:
        # 从Hugging Face加载CodeGeeX4（默认使用codegeex4-6b）
        model_name = args.arch if args.arch != "gpt2" else "zai-org/codegeex4-all-9b"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    model.eval()
    print("CodeGeeX4 loaded successfully.")

    # 主生成循环
    for index, problem in enumerate(tqdm(problems)):
        problem["solutions"] = json.loads(problem["solutions"])
        problem["input_output"] = json.loads(problem["input_output"])
        test_case = problem["input_output"]
        prompt = problem["question"]
        starter_code = problem["starter_code"] if problem["starter_code"] else None
        solutions = problem["solutions"]

        # 生成提示文本
        prompt_text, sample_sol = generate_prompt(args, test_case, prompt, solutions, tokenizer, starter_code)
        if args.debug:
            print("\nPROMPT_TEXT:")
            print(prompt_text)

        # 生成代码
        start = time.time()
        output_str = ""
        try:
            with torch.no_grad():
                # CodeGeeX4的输入处理
                inputs = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024
                ).to(device)

                # CodeGeeX4的生成参数（适配模型特性）
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024 - inputs.input_ids.shape[1],  # 改用max_new_tokens更合适
                    num_beams=args.num_beams,
                    early_stopping=True,
                    temperature=0.8,  # CodeGeeX4推荐的温度参数
                    top_p=0.95,
                    repetition_penalty=1.05,
                    do_sample=True  # CodeGeeX4生成代码时推荐使用采样
                )

                output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error generating code for problem {index + args.start}: {e}")
            output_str = ""
        end = time.time()

        # 处理生成结果
        if args.peeking == 1.0:
            output_str = sample_sol
        elif output_str:
            # 提取ANSWER部分的内容
            if "ANSWER:\n" in output_str:
                output_str = output_str.split("ANSWER:\n")[1]

        # 保存生成的代码
        gpt_codes[index + args.start] = output_str

        if args.debug:
            print(f"Generation time: {end - start:.2f}s")
            print(f"Generated output string:")
            print(output_str)
            print("-" * 80)

    # 保存所有生成的代码
    with open(codes_loc, "w", encoding="utf-8") as f:
        json.dump(gpt_codes, f, ensure_ascii=False, indent=2)
    print(f"Generated codes saved to {codes_loc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CodeGeeX4 model to generate Python code.")
    parser.add_argument("--arch")
    parser.add_argument("-t", "--test_loc", default="~/apps/data_split/test.json", type=str,
                        help="path to the test folder (unused for codeparrot/apps dataset)")
    parser.add_argument("-r", "--root", default="../", type=str, help="where the data is stored (unused)")
    parser.add_argument("-l", "--load", default="", type=str,
                        help="path to local CodeGeeX4 model directory")
    parser.add_argument("--peeking", default=0.0, type=float,
                        help="fraction of solution to include in prompt (0.0-1.0)")
    parser.add_argument("--num-beams", default=5, type=int,
                        help="number of beams for beam search")
    parser.add_argument("-s", "--start", default=0, type=int,
                        help="start index of problems to process")
    parser.add_argument("-e", "--end", default=None, type=int,
                        help="end index of problems to process")
    parser.add_argument("-i", "--index", default=None, type=int,
                        help="single problem index to process")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="enable debug mode")
    parser.add_argument("--split", type=str, default="test",
                        help="What split to use (train/valid/test)")
    parser.add_argument("--save", type=str, default="./results_codegeex4",
                        help="directory to save generated codes")

    args = parser.parse_args()

    main(args)