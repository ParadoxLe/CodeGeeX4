# 纯抽象接口，只定义模型必须实现的方法，不依赖任何外部库
from abc import ABC, abstractmethod
from typing import Optional, Dict

class BaseCodeModel(ABC):
    @abstractmethod
    def __init__(self, config: Optional[Dict] = None):
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """输入prompt，返回生成的代码字符串"""
        pass