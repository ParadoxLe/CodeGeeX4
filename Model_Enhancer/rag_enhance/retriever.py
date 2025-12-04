import numpy as np
from typing import List, Dict
from .knowledge_base import KnowledgeBase


class Retriever:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base  # 关联知识库

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """根据查询检索最相关的片段"""
        if len(self.kb) == 0:
            return []  # 知识库为空时返回空

        # 生成查询向量
        query_embedding = self.kb.embedder.encode([query])[0].astype(np.float32)
        # 检索top_k个最相似的片段（返回距离和索引）
        distances, indices = self.kb.index.search(np.array([query_embedding]), top_k)

        # 整理结果（按相似度排序，距离越小越相似）
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.kb.texts):  # 避免索引越界
                results.append({
                    "text": self.kb.texts[idx]["text"],
                    "metadata": self.kb.texts[idx]["metadata"],
                    "distance": float(distance)  # 相似度距离（供参考）
                })
        return results

    def build_prompt_with_context(self, query: str, top_k: int = 3) -> str:
        """构建带检索上下文的提示词"""
        retrieved = self.retrieve(query, top_k=top_k)
        if not retrieved:
            return query  # 无检索结果时直接返回原始查询

        # 拼接检索到的上下文
        context = "\n\n".join([f"相关参考：\n{r['text']}" for r in retrieved])
        return f"根据以下参考信息，解决用户的问题：\n{context}\n\n用户问题：{query}"