import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from typing import Optional, Dict

# 导入DFG、CFG、AST增强函数（均来自dfg_enhance目录）
from .gnn_enhance import (
    get_code_enhance_prompt,  # DFG增强
    get_control_flow_prompt,  # CFG增强
    get_ast_prompt  # AST增强
)


class EnhanceModel_GNN:
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.model_path = config.get("model_path", "zai-org/codegeex4-all-9b")
        print(f"加载独立模型：{self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True, device_map="auto"
        )
        self.default_temp = 0.5
        self.max_seq_len = self.tokenizer.model_max_length

    def generate(self, prompt: str, **kwargs) -> str:
        # 从prompt或kwargs中获取代码片段
        code_snippet = kwargs.get("code_snippet", prompt)

        # 分别生成三种增强提示词
        dfg_prompt = get_code_enhance_prompt(code_snippet)
        cfg_prompt = get_control_flow_prompt(code_snippet)
        ast_prompt = get_ast_prompt(code_snippet)

        # 整合增强信息（过滤空值）
        enhance_parts = [p for p in [dfg_prompt, cfg_prompt, ast_prompt] if p]
        final_prompt = f"{prompt}\n\n" + "\n\n".join(enhance_parts) if enhance_parts else prompt

        # 提取生成参数
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", self.default_temp)
        top_p = kwargs.get("top_p", 0.92)
        repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        do_sample = kwargs.get("do_sample", True)
        eos_token_id = kwargs.get("eos_token_id", None)
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        pad_token_id = kwargs.get("pad_token_id", None)

        # 编码
        inputs = self.tokenizer(
            final_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len - max_new_tokens
        ).to(self.model.device)

        # 生成
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
