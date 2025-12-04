"""自我反思迭代优化工具"""
from typing import Callable


def reflect_and_optimize(
        prompt: str,
        initial_code: str,
        generate_func: Callable[[str, dict], str],
        rounds: int = 2,
        **generate_kwargs
) -> str:
    """
    多轮自我反思优化代码

    参数：
        prompt: 原始问题提示词
        initial_code: 初始生成的代码
        generate_func: 模型生成函数（如model.generate）
        rounds: 反思轮次
        generate_kwargs: 生成参数（如temperature、max_new_tokens等）
    返回：
        优化后的代码
    """
    current_code = initial_code
    for round in range(rounds):
        # 构建本轮反思提示词
        from .reflection_prompt import build_reflection_prompt
        reflection_prompt = build_reflection_prompt(prompt, current_code)

        # 调用模型生成优化后的代码
        current_code = generate_func(reflection_prompt, **generate_kwargs)
        print(f"完成第 {round + 1} 轮自我反思优化")

    return current_code