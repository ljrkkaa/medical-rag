# kb_factory.py
import os, atexit
from functools import lru_cache
from .KnowledgeBase import MedicalHybridKnowledgeBase
from ..config.models import AppConfig
import json


@lru_cache(maxsize=None)
def _kb_singleton(pid: int, config_str: str):
    """按 (pid, 配置) 缓存，保证每个进程只创建一次。"""
    # 1) 还原配置：调用方传入的是可哈希字符串，这里转回 dict
    config_dict = json.loads(config_str)
    # 2) 用 Pydantic 做结构化校验与类型转换，得到 AppConfig
    config = AppConfig.model_validate(config_dict)
    # 3) 创建知识库对象（内部会初始化 Milvus client 与 embedding）
    kb = MedicalHybridKnowledgeBase(config)
    # 4) 注册进程退出钩子：尽量优雅关闭 Milvus 连接
    #    getattr(..., "close", lambda: None) 的目的是兼容没有 close 方法的客户端实现
    atexit.register(lambda: getattr(kb.client, "close", lambda: None)())
    return kb

# 严格说这不是“全局唯一单例”，而是进程内、按配置维度的单例（更准确叫 multiton）
def get_kb(config: dict):
    # 以“当前进程 PID + 配置内容”作为缓存键：
    # - 同一进程、同一配置：复用同一个知识库实例
    # - 不同进程：各自持有独立实例（避免跨进程共享连接）
    # sort_keys=True 保证相同配置字典生成稳定字符串，避免键顺序导致缓存失效
    return _kb_singleton(os.getpid(), json.dumps(config, sort_keys=True))
