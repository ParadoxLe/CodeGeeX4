# 让3个验证脚本能快速加载你的模型，不用关心内部实现
from Enhance_model import EnhanceModel
from typing import Optional, Dict


def load_code_model(config: Optional[Dict] = None) -> EnhanceModel:
    """统一加载接口，给外部脚本调用"""
    return EnhanceModel(config=config)
