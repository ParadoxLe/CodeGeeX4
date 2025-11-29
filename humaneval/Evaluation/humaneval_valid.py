import json
import os
import jsonlines
from typing import List, Dict
import warnings
import textwrap
import re

# 屏蔽HuggingFace的FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# 设置镜像源（需在加载模型前执行）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.evaluation import evaluate_functional_correctness  # 官方评估函数

# ... 其他 import ...
import jsonlines
from code_refiner import CodeRefiner  # <--- 新增
from code_gnn_reranker import GNNReranker  # <--- 之前写的 GNN

# ================= 配置参数 =================
model_path = "zai-org/codegeex4-all-9b"
dataset_name = "evalplus/humanevalplus"
output_dir = "./humaneval_results"  # 统一保存文件夹
output_file = "humaneval_candidates_with_problem.jsonl"  # 带问题标注的候选解文件
k_values = [1, 10, 100]  # 要计算的 pass@k
max_new_tokens = 1024  # 每个候选解的最大长度
temperature = 0.2  # 采样温度（生成多个解需要开启采样）
top_p = 0.95  # 核采样
batch_size = 8  # 批量生成（根据GPU显存调整）

# 创建输出文件夹（确保文件夹存在）
os.makedirs(output_dir, exist_ok=True)
final_output_path = os.path.join(output_dir, output_file)  # 最终保存路径


# ================= 核心功能函数 =================
def load_model_and_tokenizer(model_path: str):
    """加载模型和Tokenizer"""
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"  # 左padding更适合LLM生成
    )
    # 补充pad_token（如果模型没有）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"  # 自动选择高精度 dtype（如FP16/FP8）
    )
    model.eval()  # 推理模式
    return tokenizer, model


def generate_candidates(
        tokenizer, model, prompt: str, num_candidates: int, batch_size: int
) -> List[str]:
    """生成指定数量的候选解（修复批量生成溢出bug）"""
    candidates = []
    remaining = num_candidates

    # 构建对话模板
    messages = [{'role': 'user', 'content': f"""Write a solution to the following problem:
```python
{prompt}
```"""}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    while remaining > 0:
        current_batch_size = min(batch_size, remaining)

        # 仅用 num_return_sequences 控制批量生成数量
        outputs = model.generate(
            inputs,  # 单个输入，不重复
            max_new_tokens=max_new_tokens,
            do_sample=True,  # 必须开启采样才能生成不同解
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=current_batch_size,  # 1个输入生成 current_batch_size 个解
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1  # 减轻重复生成
        )

        # 解码并过滤特殊token
        for output in outputs:
            candidate = tokenizer.decode(
                output[len(inputs[0]):],  # 跳过输入部分
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            # 简单清理：只保留函数定义部分（避免多余文本）
            candidate = clean_candidate(candidate)
            if candidate:
                candidates.append(candidate)

        remaining -= current_batch_size
        print(f"Generated {len(candidates)}/{num_candidates} candidates")

    return candidates[:num_candidates]  # 确保最终只返回目标数量


def format_oneliner(code: str) -> str:
    """
    专门修复被压缩成一行的代码。
    例如: "for i in ...: if ...: return True return False"
    """
    # 1. 核心逻辑：在冒号 : 后面如果紧跟关键字 (for, if, return 等)，插入换行
    # Group 1: :
    # Group 2: 关键字
    pattern_colon = r'(:)\s+(for|while|if|elif|else|return|try|except|with|def|class)'
    code = re.sub(pattern_colon, r'\1\n\2', code)

    # 2. 处理 return/break/continue 这种结束性语句
    # 如果它们出现在行中间（前面有空格），说明需要换行
    # 例如 "... return True return False" -> "... return True\nreturn False"
    pattern_statement = r'\s+(return|break|continue|yield|raise)'
    code = re.sub(pattern_statement, r'\n\1', code)

    # 3. 重新构建缩进
    # 简单的缩进状态机
    lines = code.split('\n')
    formatted_lines = []
    indent_level = 0
    indent_unit = "    "  # 4个空格

    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue

        # 简单的启发式：如果是 else/elif，先回退一级
        if line.startswith(("else", "elif", "except", "finally")):
            indent_level = max(0, indent_level - 1)

        # 特殊修正：如果上一行是 return，且当前行也是 return，
        # 或者当前行是最后一行且是 return，通常意味着它属于外层
        # 这是一个针对 HumanEval 的特定优化
        if line.startswith("return") and i > 0 and lines[i - 1].strip().startswith("return"):
            # 连续两个 return，第二个肯定在外层，强制回退缩进
            indent_level = max(0, indent_level - 2)  # 回退两级通常比较安全

        # 添加缩进
        formatted_lines.append((indent_unit * indent_level) + line)

        # 计算下一行的缩进
        if line.endswith(":"):
            indent_level += 1
        elif line.startswith("return ") or line == "return":
            # 遇到 return，理论上缩进应该减少，但在 Python 语法解析中很难确定减少几级
            # 这里不做自动减少，依靠上面的 "连续 return" 规则来修补
            pass

    return "\n".join(formatted_lines)


def clean_candidate(candidate: str) -> str:
    # 1. 预处理：如果包含 ```，去掉外层
    if "```" in candidate:
        candidate = candidate.split("```")[-2] if len(candidate.split("```")) > 2 else candidate
        if candidate.strip().startswith("python"):
            candidate = candidate.strip()[6:]

    # 2. 去掉 Docstring
    if '"""' in candidate:
        parts = candidate.split('"""')
        if len(parts) >= 3:
            candidate = parts[-1]
    elif "'''" in candidate:
        parts = candidate.split("'''")
        if len(parts) >= 3:
            candidate = parts[-1]

    candidate = candidate.strip()

    # 3. 【关键修复】检测是否为单行代码
    # 如果代码长度大于50字符，且没有换行符，或者包含 "for " 且没有换行
    if "\n" not in candidate and len(candidate) > 20:
        return format_oneliner(candidate)

    # 4. 如果本身就是多行，保留原有逻辑，做标准 Dedent
    lines = candidate.split("\n")
    # ... (此处省略常规多行处理逻辑，因为你的问题主要是单行) ...
    # 如果你确认输入大部分是单行，直接返回 format_oneliner(candidate) 也行
    return format_oneliner(candidate)


def save_task_candidates(task_candidates: List[Dict], output_path: str):
    """追加保存单个任务的候选解到JSONL文件（带问题标注）"""
    # 使用追加模式（a），避免覆盖已保存的内容
    with jsonlines.open(output_path, "a") as f:
        f.write_all(task_candidates)
    print(f" Task candidates saved to: {output_path} (added {len(task_candidates)} candidates)\n")


def calculate_pass_at_k(candidates_path: str, k_values: List[int]) -> Dict:
    """使用官方工具计算pass@k"""
    print(f"\nCalculating pass@k for k={k_values}...")
    results = evaluate_functional_correctness(
        candidates_path,
        k=k_values,
        timeout=30  # 每个测试用例超时时间（秒）
    )
    return results


# ================= 主程序 =================
if __name__ == "__main__":
    #  新增：判断候选解文件是否已存在
    if os.path.exists(final_output_path):
        print(f" 已找到候选解文件：{final_output_path}")
        print("直接跳过生成步骤，开始评估...\n")
        # 加载数据集（仅用于计算结果时显示任务数量，不用加载模型）
        dataset = load_dataset(dataset_name, split="test")
        total_tasks = len(dataset)

    else:
        #  文件不存在时，才执行原来的「加载模型+生成候选解」逻辑
        # 1. 加载模型、Tokenizer和数据集
        tokenizer, model = load_model_and_tokenizer(model_path)
        # 初始化 Refiner (传入刚加载的模型，不需要重新加载)
        refiner = CodeRefiner(tokenizer, model)  # <--- 新增
        dataset = load_dataset(dataset_name, split="test")  # HumanEval+测试集
        print(f"Loaded dataset with {len(dataset)} tasks\n")
        # 2. 生成所有任务的候选解（每个任务生成后立即保存，带问题标注）
        max_candidates_per_task = max(k_values)
        total_tasks = len(dataset)

        #  初始化 GNN 重排序器
        # 注意：如果你还没有训练好的权重文件，这里它会使用随机权重运行
        # 这主要用于测试流程是否跑通。真正提升效果需要 'gnn_model.pth'
        gnn_reranker = GNNReranker(model_path="gnn_model.pth")

        # 2. 生成所有任务的候选解...
        max_candidates_per_task = max(k_values)

        for idx, data in enumerate(dataset):
            # 提取当前任务的关键信息（用于标注）
            task_id = f"HumanEval/{idx}"
            problem_prompt = data["prompt"].strip()  # 完整问题描述（函数定义+注释）
            # 提取函数名作为任务名（更易读）
            func_name = None
            for line in problem_prompt.split("\n"):
                if line.startswith("def "):
                    func_name = line.split("def ")[1].split("(")[0].strip()
                    break
            task_name = f"func_{func_name}" if func_name else f"task_{idx}"  # 如 func_fibonacci

            print(f"=== Processing task {idx + 1}/{total_tasks} ===")
            print(f"Task ID: {task_id}")
            print(f"Task Name: {task_name}")
            print(f"Problem Preview:\n{problem_prompt[:300]}..." if len(
                problem_prompt) > 300 else f"Problem:\n{problem_prompt}")

            # 生成当前任务的候选解
            candidates = generate_candidates(
                tokenizer=tokenizer,
                model=model,
                prompt=problem_prompt,
                num_candidates=max_candidates_per_task,
                batch_size=batch_size
            )

            # 【关键步骤】使用 GNN 对候选解进行重排序
            # 原理：虽然生成了 100 个，但排在第 0 位的可能是错的。
            # GNN 尝试把“长得像正确代码”的解排到最前面。
            print(f"  > Reranking {len(candidates)} candidates with GNN...")
            candidates = gnn_reranker.rerank(candidates)

            # 构建当前任务的候选解列表（带问题标注）
            task_candidates = []
            for cand_idx, candidate in enumerate(candidates):
                task_candidates.append({
                    "task_id": task_id,  # 标准任务ID（如 HumanEval/0）
                    "task_name": task_name,  # 易读任务名（如 func_fibonacci）
                    "problem_description": problem_prompt,  # 完整问题描述（方便追溯）
                    "candidate_id": f"{task_id}_candidate_{cand_idx}",  # 唯一候选解ID（如 HumanEval/0_candidate_5）
                    "candidate_index": cand_idx,  # 候选解在当前任务中的序号（0-99）
                    "completion": candidate  # 模型生成的解
                })

            # 逐任务追加保存（带问题标注）
            save_task_candidates(task_candidates, final_output_path)

    # 👇 评估逻辑不变（无论文件是否存在，都会执行）
    print("\n" + "=" * 60)
    print("Starting pass@k evaluation...")
    print("=" * 60)

    results = calculate_pass_at_k(final_output_path, k_values)

    # 格式化输出结果
    print("\n" + "=" * 50)
    print("Pass@k Results (HumanEval+)")
    print("=" * 50)
    for k in k_values:
        pass_k = results[f"pass@{k}"]
        print(f"pass@{k}: {pass_k:.4f} ({int(pass_k * total_tasks)}/{total_tasks} tasks passed)")
    print("=" * 50)

    # 保存pass@k结果到同一文件夹
    results_path = os.path.join(output_dir, "pass_at_k_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n Pass@k results saved to: {results_path}")
    print(f" 带问题标注的候选解文件：{final_output_path}")
    print(f" 所有文件存储路径：{output_dir}")
