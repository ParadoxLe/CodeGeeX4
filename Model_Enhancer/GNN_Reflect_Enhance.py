from .base_model import BaseCodeModel
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from typing import Optional, Dict

# 导入GNN增强模块
from .gnn_enhance import (
    get_code_enhance_prompt,  # DFG增强
    get_control_flow_prompt,  # CFG增强
    get_ast_prompt  # AST增强
)

# 导入自我反思模块
from .reflection_enhance import reflect_and_optimize


class GNN_Reflect_Enhance(BaseCodeModel):
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.model_path = config.get("model_path", "zai-org/codegeex4-all-9b")

        # 加载模型
        print(f"加载GNN+Reflect增强模型：{self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True, device_map="auto"
        )

        # 配置参数
        self.default_temp = 0.5
        self.max_seq_len = self.tokenizer.model_max_length
        self.reflection_rounds = config.get("reflection_rounds", 2)  # 反思轮次（可配置）

    def generate(self, prompt: str, **kwargs) -> str:
        """GNN增强+自我反思的代码生成流程"""
        # 1. 提取代码片段（优先从kwargs获取，否则使用prompt）
        code_snippet = kwargs.get("code_snippet", prompt)

        # 2. GNN增强：生成DFG/CFG/AST提示词
        dfg_prompt = get_code_enhance_prompt(code_snippet)
        cfg_prompt = get_control_flow_prompt(code_snippet)
        ast_prompt = get_ast_prompt(code_snippet)

        # 3. 整合GNN增强提示词
        enhance_parts = [p for p in [dfg_prompt, cfg_prompt, ast_prompt] if p]
        gnn_enhanced_prompt = f"{prompt}\n\n" + "\n\n".join(enhance_parts) if enhance_parts else prompt

        # 4. 生成初始代码（基于GNN增强后的提示词）
        initial_code = self._generate_single(gnn_enhanced_prompt,** kwargs)

        # 5. 自我反思优化（多轮迭代）
        optimized_code = reflect_and_optimize(
            prompt=prompt,  # 使用原始问题作为反思基准
            initial_code=initial_code,
            generate_func=self._generate_single,  # 传入单次生成函数
            rounds=self.reflection_rounds,
            **kwargs  # 透传生成参数
        )

        return optimized_code

    def _generate_single(self, prompt: str,** kwargs) -> str:
        """内部单次生成方法（供反思模块调用）"""
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", self.default_temp)
        top_p = kwargs.get("top_p", 0.92)
        repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        do_sample = kwargs.get("do_sample", True)
        eos_token_id = kwargs.get("eos_token_id", None)
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        pad_token_id = kwargs.get("pad_token_id", None)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len - max_new_tokens
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
            num_return_sequences=num_return_sequences,
            pad_token_id=pad_token_id
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()