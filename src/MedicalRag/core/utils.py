"""
工具类，创建合适的llm和embedding客户端
"""

import logging
import json
from threading import Lock
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from ..config.models import LLMConfig, DenseConfig, AppConfig
from .MultiGpuEmbeddings import MultiGpuSentenceTransformerEmbeddings
import os
import torch

logger = logging.getLogger(__name__)

_EMBEDDING_CLIENT_CACHE: dict[str, Embeddings] = {}
_EMBEDDING_CACHE_LOCK = Lock()


def _embedding_cache_key(config: DenseConfig) -> str:
    """生成 embedding 客户端缓存键（仅包含影响实例构造的字段）。"""
    key_dict = {
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "env_key_name": config.env_key_name,
        "proxy": config.proxy,
        "dimension": config.dimension,
        "multi_gpu": config.multi_gpu,
        "num_gpus": config.num_gpus,
        "encode_batch_size": config.encode_batch_size,
    }
    return json.dumps(key_dict, ensure_ascii=False, sort_keys=True)


def create_llm_client(config: LLMConfig) -> BaseChatModel:
    """创建LLM客户端"""
    if config.provider == "openai":
        kwargs = {
            "model": config.model,
            "temperature": config.temperature,
        }

        if config.env_key_name:
            kwargs["api_key"] = os.environ[config.env_key_name]
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.max_tokens:
            kwargs["max_tokens"] = config.max_tokens
        if config.proxy:
            kwargs["http_client"] = {
                "proxies": {"http": config.proxy, "https": config.proxy}
            }

        return ChatOpenAI(**kwargs)

    elif config.provider == "ollama":
        kwargs = {
            "model": config.model,
            "temperature": config.temperature,
        }

        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.max_tokens:
            kwargs["num_predict"] = config.max_tokens

        return ChatOllama(**kwargs)

    else:
        raise ValueError(f"不支持的LLM提供商: {config.provider}")


def create_embedding_client(config: DenseConfig) -> Embeddings:
    """创建嵌入客户端"""
    cache_key = _embedding_cache_key(config)
    with _EMBEDDING_CACHE_LOCK:
        cached = _EMBEDDING_CLIENT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if config.provider == "openai":
        kwargs = {"model": config.model, "dimensions": config.dimension}

        if config.env_key_name:
            kwargs["api_key"] = os.environ[config.env_key_name]
        if config.base_url:
            kwargs["base_url"] = config.base_url
            # 对 OpenAI 兼容网关（如 DashScope）避免发送 token id 列表，直接发送字符串文本。
            if "api.openai.com" not in config.base_url:
                kwargs["check_embedding_ctx_length"] = False
        if config.proxy:
            kwargs["http_client"] = {
                "proxies": {"http": config.proxy, "https": config.proxy}
            }

        client = OpenAIEmbeddings(**kwargs)
        with _EMBEDDING_CACHE_LOCK:
            _EMBEDDING_CLIENT_CACHE[cache_key] = client
        return client

    elif config.provider == "ollama":
        kwargs = {"model": config.model}
        if config.base_url:
            kwargs["base_url"] = config.base_url

        client = OllamaEmbeddings(**kwargs)
        with _EMBEDDING_CACHE_LOCK:
            _EMBEDDING_CLIENT_CACHE[cache_key] = client
        return client

    elif config.provider == "embedding":
        if (
            config.multi_gpu
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
        ):
            n = max(1, min(config.num_gpus, torch.cuda.device_count()))
            devices = [f"cuda:{i}" for i in range(n)]
            client = MultiGpuSentenceTransformerEmbeddings(
                model_name=config.model,
                normalize_embeddings=True,
                target_devices=devices,
                batch_size=config.encode_batch_size,
            )
            with _EMBEDDING_CACHE_LOCK:
                _EMBEDDING_CLIENT_CACHE[cache_key] = client
            return client

        # 使用本地torch加载embedding模型 (如 bge-m3)
        kwargs = {
            "model_name": config.model,
            "encode_kwargs": {"normalize_embeddings": True},
        }
        # 指定 device 为 cuda 或 cpu
        kwargs["model_kwargs"] = {
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

        client = HuggingFaceEmbeddings(**kwargs)
        with _EMBEDDING_CACHE_LOCK:
            _EMBEDDING_CLIENT_CACHE[cache_key] = client
        return client

    else:
        raise ValueError(f"不支持的嵌入提供商: {config.provider}")


def preload_embedding_clients(app_config: AppConfig) -> None:
    """按配置预加载常用 Embedding 客户端并写入缓存。"""
    dense_configs = [
        app_config.embedding.summary_dense,
        app_config.embedding.text_dense,
    ]
    for cfg in dense_configs:
        if getattr(cfg, "preload", False):
            try:
                create_embedding_client(cfg)
                logger.info(
                    "Embedding 预加载完成: provider=%s, model=%s",
                    cfg.provider,
                    cfg.model,
                )
            except Exception as e:
                logger.warning(
                    "Embedding 预加载失败: provider=%s, model=%s, err=%s",
                    cfg.provider,
                    cfg.model,
                    e,
                )
