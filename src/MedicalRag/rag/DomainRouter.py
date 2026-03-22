from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from ..config.models import RoutingConfig, AppConfig
from ..core.utils import create_llm_client
from ..prompts.templates import DOMAIN_ROUTER_PROMPT
from .CentroidBuilder import build_and_save_centroids

logger = logging.getLogger(__name__)


class DomainRouter:
    """可复用的领域路由器：支持 centroid / llm 两种策略。"""

    def __init__(
        self,
        embedding: Embeddings,
        default_llm: BaseChatModel,
        routing_config: RoutingConfig,
        app_config: Optional[AppConfig] = None,
    ):
        self.embedding = embedding
        self.config = routing_config
        self.app_config = app_config
        self._default_llm = default_llm
        self.llm: Optional[BaseChatModel] = None

        self.domain_centroids: dict[str, np.ndarray] = {}
        self._load_domain_centroids(routing_config.centroid_file)

    def _get_llm(self) -> BaseChatModel:
        """按需初始化路由 LLM，避免无谓开销。"""
        if self.llm is None:
            self.llm = (
                create_llm_client(self.config.llm)
                if self.config.llm
                else self._default_llm
            )
        return self.llm

    def _preferred_domain(self) -> str:
        return (self.config.preferred_domain or "").strip()

    def _routing_domains_from_centroids(self) -> list[str]:
        """用于路由决策的 domain 列表（排除 preferred_domain）。"""
        preferred = self._preferred_domain()
        return [d for d in self.domain_centroids.keys() if d != preferred]

    def _routing_fallback_domains(self) -> list[str]:
        """用于路由决策的 fallback 列表（排除 preferred_domain）。"""
        preferred = self._preferred_domain()
        return [d for d in self.config.fallback_domains if d != preferred]

    def _ensure_centroid_file(self, centroid_file: str) -> bool:
        p = Path(centroid_file)
        if p.exists():
            # 已有文件时，检查是否缺少新 domain（例如新增了 data/processed/*.jsonl）
            if self.config.auto_build_centroid and self.app_config is not None:
                try:
                    processed_dir = Path(self.config.centroid_processed_dir)
                    preferred = self._preferred_domain()
                    expected_domains = {
                        fp.stem for fp in processed_dir.glob("*.jsonl") if fp.is_file()
                    }
                    if preferred:
                        expected_domains.discard(preferred)
                    if expected_domains:
                        with np.load(p) as data:
                            existing_domains = set(data.files)
                        if preferred:
                            existing_domains.discard(preferred)
                        missing = sorted(expected_domains - existing_domains)
                        if missing:
                            logger.info(
                                "检测到 centroid 缺少新 domain，自动重建: missing=%s",
                                missing,
                            )
                            _ = build_and_save_centroids(
                                config=self.app_config,
                                mode=self.config.centroid_build_mode,
                                processed_dir=processed_dir,
                                output_file=p,
                                batch_size=self.config.centroid_batch_size,
                            )
                except Exception as e:
                    logger.warning("检查/重建 centroid 失败，继续使用现有文件: %s", e)

            logger.info("检测到 centroid 文件，跳过初始化构建: %s", centroid_file)
            return True

        if not self.config.auto_build_centroid:
            logger.warning("centroid 文件不存在且已关闭自动构建: %s", centroid_file)
            return False
        if self.app_config is None:
            logger.warning(
                "centroid 文件不存在，且 DomainRouter 未收到 app_config，无法自动构建"
            )
            return False

        try:
            logger.info(
                "centroid 文件不存在，开始自动构建: mode=%s, processed_dir=%s, batch_size=%s",
                self.config.centroid_build_mode,
                self.config.centroid_processed_dir,
                self.config.centroid_batch_size,
            )
            _ = build_and_save_centroids(
                config=self.app_config,
                mode=self.config.centroid_build_mode,
                processed_dir=Path(self.config.centroid_processed_dir),
                output_file=p,
                batch_size=self.config.centroid_batch_size,
            )
            return p.exists()
        except Exception as e:
            logger.warning("自动构建 centroid 失败: %s", e)
            return False

    def _load_domain_centroids(self, centroid_file: str):
        if not self._ensure_centroid_file(centroid_file):
            logger.warning("centroid 文件不存在: %s", centroid_file)
            return
        p = Path(centroid_file)
        try:
            data = np.load(p)
            self.domain_centroids = {k: data[k].astype("float32") for k in data.files}
            logger.info(
                "DomainRouter 加载 centroid 成功: %s",
                list(self.domain_centroids.keys()),
            )
        except Exception as e:
            logger.warning("DomainRouter 加载 centroid 失败: %s", e)
            self.domain_centroids = {}

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _route_by_centroid(
        self, query: str, top_k_override: Optional[int] = None
    ) -> Optional[list[str]]:
        routing_domains = self._routing_domains_from_centroids()
        if not routing_domains:
            return None
        try:
            q = np.array(self.embedding.embed_query(query), dtype="float32")
            scored = [
                (domain, self._cosine(q, c))
                for domain in routing_domains
                for c in [self.domain_centroids[domain]]
            ]
            scored.sort(key=lambda x: -x[1])
            top_k_value = top_k_override if top_k_override else self.config.top_k
            top_k = max(1, top_k_value)
            candidates = [d for d, _ in scored[:top_k]]
            return candidates if candidates else None
        except Exception as e:
            logger.warning("centroid 路由失败: %s", e)
            return None

    def _route_by_llm(self, query: str) -> Optional[str]:
        try:
            domains = self._routing_domains_from_centroids() or self._routing_fallback_domains()
            options = ", ".join(domains + ["Other"])
            prompt = PromptTemplate.from_template(DOMAIN_ROUTER_PROMPT)
            chain = prompt | self._get_llm() | StrOutputParser()
            pred = chain.invoke({"question": query, "options": options}).strip()
            pred = pred.replace("`", "").replace("。", "").split()[0]
            return pred if pred in domains else None
        except Exception as e:
            logger.warning("llm 路由失败: %s", e)
            return None

    def _merge_with_preferred_domain(
        self, routed: Optional[Union[str, list[str]]]
    ) -> Optional[Union[str, list[str]]]:
        """将 preferred_domain 放在首位，并去重。"""
        if not self.config.always_include_preferred_domain:
            return routed

        preferred = (self.config.preferred_domain or "").strip()
        if not preferred:
            return routed

        domains: list[str] = [preferred]
        if isinstance(routed, str):
            r = routed.strip()
            if r and r != preferred:
                domains.append(r)
        elif isinstance(routed, list):
            for d in routed:
                s = str(d).strip()
                if s and s not in domains:
                    domains.append(s)

        return domains

    def route(self, inputs: dict) -> Optional[Union[str, list[str]]]:
        # 手动指定优先级最高
        manual_domain = inputs.get("domain")
        if manual_domain:
            if isinstance(manual_domain, list):
                return [str(x) for x in manual_domain if str(x).strip()]
            return str(manual_domain)

        method = str(inputs.get("routing_method") or self.config.method).lower()
        top_k = inputs.get("routing_top_k")
        top_k_override = top_k if isinstance(top_k, int) and top_k > 0 else None
        query = str(inputs.get("input", ""))

        if method == "llm":
            routed = self._route_by_llm(query)
            if routed:
                return self._merge_with_preferred_domain(routed)
            # llm 失败时回退 centroid
            routed = self._route_by_centroid(query, top_k_override=top_k_override)
            return self._merge_with_preferred_domain(routed)

        # 默认 centroid；失败时回退 llm
        routed = self._route_by_centroid(query, top_k_override=top_k_override)
        if routed:
            return self._merge_with_preferred_domain(routed)
        routed = self._route_by_llm(query)
        return self._merge_with_preferred_domain(routed)

    def as_runnable(self) -> Runnable:
        return RunnableLambda(self.route).with_config(run_name="route_domain")
