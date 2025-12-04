from .base_model import BaseCodeModel
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from typing import Optional, Dict

# 导入自我反思模块
from .reflection_enhance import reflect_and_optimize


class EnhanceModel_Reflect(BaseCodeModel):
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.model_path = config.get("model_path", "zai-org/codegeex4-all-9b")

        # 加载模型
        print(f"加载独立模型：{self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True, device_map="auto"
        )

        # 配置参数
        self.default_temp = 0.5
        self.max_seq_len = self.model.config.max_length  # 补充序列长度定义
        self.reflection_rounds = config.get("reflection_rounds", 2)  # 反思轮次（可配置）

    def generate(self, prompt: str, **kwargs) -> str:
        """带自我反思的代码生成：调用反思模块进行多轮优化"""
        # 1. 生成初始代码
        initial_code = self._generate_single(prompt, **kwargs)

        # 2. 调用反思模块进行多轮优化
        optimized_code = reflect_and_optimize(
            prompt=prompt,
            initial_code=initial_code,
            generate_func=self._generate_single,  # 传入单次生成函数
            rounds=self.reflection_rounds,
            **kwargs  # 透传生成参数（如temperature等）
        )

        return optimized_code

    def _generate_single(self, prompt: str, **kwargs) -> str:
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