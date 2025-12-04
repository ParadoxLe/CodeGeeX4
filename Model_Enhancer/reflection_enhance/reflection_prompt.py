"""自我反思提示词模板定义"""

REFLECTION_TEMPLATE = """
你需要根据以下要求优化生成的代码：
1. 检查代码是否完全符合原始问题的所有需求
2. 验证代码是否存在语法错误、逻辑漏洞或边界条件缺失
3. 确保代码可运行，且测试用例能够通过
4. 优化代码可读性（如补充必要注释、简化冗余逻辑）

原始问题：
{prompt}

已生成的代码：
{generated_code}

请输出优化后的完整代码（仅返回代码内容，不要额外说明）：
"""

def build_reflection_prompt(prompt: str, generated_code: str) -> str:
    """构建反思提示词"""
    return REFLECTION_TEMPLATE.format(
        prompt=prompt,
        generated_code=generated_code
    )