# 让3个验证脚本能快速加载你的模型，不用关心内部实现
from .Enhance_model import EnhanceModel
from .Enhance_model_DFG import EnhanceModel_DFG
from typing import Optional, Dict


def load_code_model(config: Optional[Dict] = None) -> EnhanceModel:
    """统一加载接口，给外部脚本调用"""
    return EnhanceModel(config=config)


def load_code_model_DFG(config: Optional[Dict] = None) -> EnhanceModel:
    """统一加载接口，给外部脚本调用"""
    return EnhanceModel_DFG(config=config)

