import json
import os
import jsonlines
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # 屏蔽HuggingFace的FutureWarning
# 设置镜像源（需在加载模型前执行）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.evaluation import evaluate_functional_correctness  # 官方评估函数

# ================= 配置参数 =================
model_path = "zai-org/codegeex4-all-9b"
dataset_name = "evalplus/humanevalplus"
output_dir = "./humaneval_results"  # 统一保存文件夹
output_file = "humaneval_candidates_with_problem.jsonl"  # 带问题标注的候选解文件
k_values = [1, 10, 100]  # 要计算的 pass@k
max_new_tokens = 1024  # 每个候选解的最大长度
temperature = 0.2  # 采样温度（生成多个解需要开启采样）
top_p = 0.95
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

        # 修复：移除 inputs.repeat，仅用 num_return_sequences 控制批量生成数量
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


def clean_candidate(candidate: str) -> str:
    """清理候选解：提取函数定义，去除多余内容"""
    # 找到函数定义的开始（def 或 class）
    lines = [line.strip() for line in candidate.split("\n") if line.strip()]
    func_lines = []
    in_func = False
    indent_level = 0

    for line in lines:
        if line.startswith(("def ", "class ")):
            in_func = True
            func_lines.append(line)
            # 计算缩进级别（假设用4个空格）
            indent_level = len(line) - len(line.lstrip())
        elif in_func:
            # 如果当前行缩进 >= 函数定义的缩进，属于函数内容
            current_indent = len(line) - len(line.lstrip())
            if current_indent >= indent_level or line.startswith(("return", "if", "for", "while", "with", "try")):
                func_lines.append(line)
            else:
                # 缩进变小，说明函数结束
                break

    return "\n".join(func_lines) if func_lines else candidate


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
    # 👇 新增：判断候选解文件是否已存在
    if os.path.exists(final_output_path):
        print(f"📄 已找到候选解文件：{final_output_path}")
        print("直接跳过生成步骤，开始评估...\n")
        # 加载数据集（仅用于计算结果时显示任务数量，不用加载模型）
        dataset = load_dataset(dataset_name, split="test")
        total_tasks = len(dataset)
    else:
        # 👇 文件不存在时，才执行原来的「加载模型+生成候选解」逻辑
        # 1. 加载模型、Tokenizer和数据集
        tokenizer, model = load_model_and_tokenizer(model_path)
        dataset = load_dataset(dataset_name, split="test")  # HumanEval+测试集
        print(f"Loaded dataset with {len(dataset)} tasks\n")

        # 2. 生成所有任务的候选解（每个任务生成后立即保存，带问题标注）
        max_candidates_per_task = max(k_values)
        total_tasks = len(dataset)

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

            # 构建当前任务的候选解列表（带问题标注）
            task_candidates = []
            for cand_idx, candidate in enumerate(candidates):
                task_candidates.append({
                    "task_id": task_id,  # 标准任务ID（如 HumanEval/0）
                    "task_name": task_name,  # 易读任务名（如 func_fibonacci）
                    "problem_description": problem_prompt,  # 完整问题描述（方便追溯）
                    "candidate_id": f"{task_id}_candidate_{cand_idx}",  # 唯一候选解ID（如 HumanEval/0_candidate_5）
                    "candidate_index": cand_idx,  # 候选解在当前任务中的序号（0-99）
                    "solution": candidate  # 模型生成的解
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