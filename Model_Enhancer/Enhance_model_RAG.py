from .base_model import BaseCodeModel
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)  # 屏蔽HuggingFace的FutureWarning
from typing import Optional, Dict
# ==========改动一，添加==========
# 导入RAG模块（新增）
from .rag_enhance import KnowledgeBase, Retriever


class EnhanceModel(BaseCodeModel):
    def __init__(self, config: Optional[Dict] = None):
        # 独立配置
        config = config or {}
        self.model_path = config.get("model_path", "zai-org/codegeex4-all-9b")  # 你的模型路径
        print("现在使用的是RAG模块")

        # ==========改动二，添加==========
        # 初始化RAG
        self.kb = KnowledgeBase(kb_dir=config.get("rag_kb_dir", "RAG/knowledge_base"))
        self.retriever = Retriever(self.kb)
        print(f"RAG知识库初始化完成，当前片段数：{len(self.kb)}")

        # 加载模型（你可以任意修改这里的逻辑）
        print(f"加载独立模型：{self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True, device_map="auto"
        )
        # 默认参数
        self.default_temp = 0.5

        # ==========改动三，添加==========
        # 新增：获取模型最大序列长度
        self.max_seq_len = self.model.config.max_length

    # ==========改动四，添加==========
    def add_to_knowledge(self, text: str, metadata: Optional[Dict] = None):
        """向知识库添加文本（供外部调用）"""
        self.kb.add_text(text, metadata)

    def generate(self, prompt: str, **kwargs) -> str:
        """参数透传：优先使用 NCB 传入的参数，没有则用默认值"""
        # 新增：通过RAG增强提示词
        enhanced_prompt = self.retriever.build_prompt_with_context(prompt, top_k=3)
        # 从 kwargs 中提取参数
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", self.default_temp)
        top_p = kwargs.get("top_p", 0.92)  # 默认值，会被传入的值覆盖
        repetition_penalty = kwargs.get("repetition_penalty", 1.0)  # 同理
        do_sample = kwargs.get("do_sample", True)
        eos_token_id = kwargs.get("eos_token_id", None)
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        pad_token_id = kwargs.get("pad_token_id", None)

        # 编码
        inputs = self.tokenizer(
            # ==========改动五，替换prompt==========
            enhanced_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len - max_new_tokens
        ).to(self.model.device)

        # 生成：传入所有提取的参数
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,  # 透传 NCB 中的 top_p
            repetition_penalty=repetition_penalty,  # 透传其他参数
            do_sample=do_sample,  # 透传采样开关
            eos_token_id=eos_token_id,
            num_return_sequences=num_return_sequences,
            pad_token_id=pad_token_id
        )

        # 解码
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
