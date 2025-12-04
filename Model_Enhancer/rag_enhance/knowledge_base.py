import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import faiss  # 向量数据库
from sentence_transformers import SentenceTransformer
from .utils import split_text, read_jsonl, write_jsonl, clean_code_snippet


class KnowledgeBase:
    def __init__(self, kb_dir: str = "knowledge_base", embed_model: str = "all-MiniLM-L6-v2"):
        """初始化知识库（存储在Model_Enhancer目录下）"""
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在

        # 向量索引和文本存储路径
        self.index_path = self.kb_dir / "faiss_index.bin"
        self.texts_path = self.kb_dir / "texts.jsonl"

        # 加载嵌入模型（生成文本向量）
        self.embedder = SentenceTransformer(embed_model)
        self.dimension = self.embedder.get_sentence_embedding_dimension()  # 向量维度

        # 加载或初始化向量索引和文本数据
        self.index = self._load_or_init_index()
        self.texts = self._load_or_init_texts()  # 存储格式: [{"text": "...", "metadata": {...}}, ...]

    def _load_or_init_index(self) -> faiss.Index:
        """加载现有索引或初始化新索引"""
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        else:
            # 使用L2距离的扁平索引（适合小规模知识库）
            return faiss.IndexFlatL2(self.dimension)

    def _load_or_init_texts(self) -> List[Dict]:
        """加载现有文本片段或初始化空列表"""
        if self.texts_path.exists():
            return read_jsonl(str(self.texts_path))
        else:
            return []

    def add_text(self, text: str, metadata: Optional[Dict] = None, chunk_size: int = 500):
        """添加文本到知识库（自动分割为片段）"""
        metadata = metadata or {}
        # 分割文本为片段（适合长文档/代码）
        chunks = split_text(text, chunk_size=chunk_size)
        for chunk in chunks:
            # 生成向量
            embedding = self.embedder.encode([chunk])[0].astype(np.float32)
            # 添加到向量索引
            self.index.add(np.array([embedding]))
            # 保存文本和元数据
            self.texts.append({
                "text": chunk,
                "metadata": metadata
            })
        # 持久化更新
        self._save()

    def add_code_file(self, file_path: str, language: str = "python"):
        """添加代码文件到知识库（自动清洗和分割）"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"代码文件不存在: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        # 清洗代码（去除注释等）
        cleaned_code = clean_code_snippet(code)
        # 添加到知识库，元数据标记语言类型
        self.add_text(cleaned_code, metadata={"type": "code", "language": language, "source": file_path})

    def _save(self):
        """持久化索引和文本数据"""
        faiss.write_index(self.index, str(self.index_path))
        write_jsonl(self.texts, str(self.texts_path))

    def __len__(self):
        """返回知识库中的片段数量"""
        return len(self.texts)