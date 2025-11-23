# 基于自我反思的迭代修复
import ast
import torch
import re


class CodeRefiner:
    """
    代码迭代修复器 (Iterative Refiner)
    核心功能：
    1. 语法检查 (Syntax Check)：利用 ast 模块检测硬性语法错误。
    2. 自我反思 (Self-Correction)：将错误信息或优化指令反馈给 LLM，要求重写。
    """

    def __init__(self, tokenizer, model, max_new_tokens=1024, temperature=0.2):
        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = model.device

    def check_syntax(self, code: str) -> str:
        """
        静态分析：检查代码是否有语法错误。
        返回：错误信息字符串，如果没有错误返回 None。
        """
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            return f"{e.msg} at line {e.lineno}"
        except Exception as e:
            return str(e)

    def build_refinement_prompt(self, problem_prompt: str, wrong_code: str, error_msg: str = None) -> list:
        """
        构建用于修复代码的 Prompt。
        区分“语法修复”和“逻辑优化”两种场景。
        """
        if error_msg:
            # 场景 A: 存在明确的语法报错
            instruction = f"""
                            The code you generated contains a syntax error: "{error_msg}".
                            Please fix the syntax error and return the corrected code.
                            """
        else:
            # 场景 B: 语法没问题，进行逻辑反思 (Self-Reflection)
            instruction = """
                            Please review the code above. 
                            1. Check for infinite loops or logical boundary errors.
                            2. Ensure all variables are defined before use.
                            3. Optimize the logic for the problem description.
                            Rewrite the solution to be correct and robust.
                            """

        content = f"""
                        ### Problem:
                        {problem_prompt}

                    ### Your Previous Attempt:
                    ```python
                    {wrong_code}
                    Instruction:
                    {instruction}
                    """
        return [{'role': 'user', 'content': content}]


    def refine(self, problem_prompt: str, candidate_code: str) -> str:
        """
        核心方法：输入问题和候选解，返回修复后的解。
        """
        print(f"    [Refiner] Analyzing candidate...")

        # 1. 静态检查 (Static Analysis)
        syntax_error = self.check_syntax(candidate_code)

        # 2. 构建反思 Prompt
        messages = self.build_refinement_prompt(problem_prompt, candidate_code, syntax_error)

        # 3. 调用 LLM 生成修复版
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,  # 修复时温度稍低，求稳
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 4. 解码
        refined_full = self.tokenizer.decode(
            outputs[0][len(inputs[0]):],
            skip_special_tokens=True
        )

        # 5. 清理代码 (复用你主程序里的逻辑，这里简单清理)
        refined_code = self._clean_code(refined_full)

        # 简单对比日志
        if syntax_error:
            print(f"    [Refiner] Fixed Syntax Error: {syntax_error}")
        else:
            print(f"    [Refiner] Performed Logical Self-Correction.")

        return refined_code


    def _clean_code(self, text: str) -> str:
        """简单的内部清理函数，去除非代码部分"""
        if "```" in text:
            parts = text.split("```")
            # 找最长的或者是包含 def 的那一段
            for part in parts:
                if "def " in part or "return " in part:
                    text = part
                    break
            if text.strip().startswith("python"):
                text = text.strip()[6:]
        return text.strip()


