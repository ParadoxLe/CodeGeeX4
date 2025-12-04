import json
import re
from typing import List, Dict

def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """将长文本分割为带重叠的片段"""
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap  # 保留重叠部分避免语义割裂
    return chunks

def read_jsonl(file_path: str) -> List[Dict]:
    """读取JSONL文件（存储知识库片段）"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]

def write_jsonl(data: List[Dict], file_path: str):
    """写入JSONL文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def clean_code_snippet(code: str) -> str:
    """清洗代码片段（去除注释、多余空行）"""
    # 去除单行注释
    code = re.sub(r"#.*?$", "", code, flags=re.MULTILINE)
    # 去除多行注释
    code = re.sub(r'"""(.*?)"""', "", code, flags=re.DOTALL)
    # 去除多余空行
    code = re.sub(r"\n+", "\n", code).strip()
    return code