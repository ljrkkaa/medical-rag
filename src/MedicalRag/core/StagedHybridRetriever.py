from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..config.models import SearchRequest, StagedRetrievalConfig
from ..rag.DomainRouter import DomainRouter
from .ColBERTReranker import ColBERTReranker
from .KnowledgeBase import MedicalHybridKnowledgeBase

logger = logging.getLogger(__name__)


class StagedHybridRetriever(BaseRetriever):
    """分阶段检索器：路由 -> 分域检索 -> merge -> 粗排 -> ColBERT。"""

    knowledge_base: MedicalHybridKnowledgeBase
    search_config: SearchRequest
    domain_router: DomainRouter
    staged_config: StagedRetrievalConfig
    colbert_reranker: Optional[ColBERTReranker] = None

    def __init__(
        self,
        knowledge_base: MedicalHybridKnowledgeBase,
        search_config: SearchRequest,
        domain_router: DomainRouter,
        staged_config: StagedRetrievalConfig,
    ):
        super().__init__(
            **{
                "knowledge_base": knowledge_base,
                "search_config": search_config,
                "domain_router": domain_router,
                "staged_config": staged_config,
            }
        )

        if staged_config.colbert.enabled:
            try:
                self.colbert_reranker = ColBERTReranker(
                    model_name=staged_config.colbert.model_name,
                    max_length=staged_config.colbert.max_length,
                    top_k=staged_config.colbert.top_k,
                    batch_size=staged_config.colbert.batch_size,
                    device=staged_config.colbert.device,
                )
            except Exception as e:
                logger.warning("ColBERT初始化失败，回退到粗排序结果: %s", e)
                self.colbert_reranker = None

    @staticmethod
    def _ensure_domains(domain_value: Optional[Union[str, List[str]]]) -> List[str]:
        if not domain_value:
            return []
        if isinstance(domain_value, list):
            return [str(x) for x in domain_value if str(x).strip()]
        return [str(domain_value)]

    @staticmethod
    def _distance_to_score(distance: Any) -> float:
        if distance is None:
            return 0.0
        try:
            d = float(distance)
        except Exception:
            return 0.0
        return 1.0 / (1.0 + math.exp(d))

    def _with_domain_request(
        self, query: str, domain: str, limit: int
    ) -> SearchRequest:
        req = self.search_config.model_copy(deep=True)
        req.query = query
        req.domain = domain
        req.limit = limit
        for item in req.requests:
            item.limit = max(item.limit, limit)
        return req

    def _coarse_rank(
        self,
        grouped_results: Dict[str, List[Document]],
        routed_domains: List[str],
    ) -> List[Document]:
        cfg = self.staged_config.coarse_rank
        domain_rank_map = {d: i for i, d in enumerate(routed_domains)}

        merged: Dict[str, Document] = {}
        for domain, docs in grouped_results.items():
            domain_idx = domain_rank_map.get(domain, len(routed_domains))
            domain_prior = 1.0 / (domain_idx + 1.0)

            for rank_idx, doc in enumerate(docs):
                md = dict(doc.metadata) if doc.metadata else {}
                rank_score = 1.0 / (rank_idx + 1.0)
                dist_score = self._distance_to_score(md.get("distance"))
                fusion_score = (
                    cfg.weight_distance * dist_score
                    + cfg.weight_rank * rank_score
                    + cfg.weight_domain_prior * domain_prior
                )
                md["retrieval_rank"] = rank_idx + 1
                md["routed_domain_rank"] = domain_idx + 1
                md["coarse_score"] = fusion_score

                doc_key = str(
                    md.get("pk") or f"{domain}-{rank_idx}-{hash(doc.page_content)}"
                )
                new_doc = Document(page_content=doc.page_content, metadata=md)

                old = merged.get(doc_key)
                if old is None or old.metadata.get("coarse_score", 0.0) < fusion_score:
                    merged[doc_key] = new_doc

        coarse_docs = sorted(
            merged.values(),
            key=lambda x: x.metadata.get("coarse_score", 0.0),
            reverse=True,
        )
        return coarse_docs[: self.staged_config.merge_limit]

    def _get_relevant_documents(self, inputs: dict) -> Dict[str, Any]:
        t0 = time.time()
        query = str(inputs.get("input", ""))

        manual_domain = inputs.get("domain")
        routed_domains = self._ensure_domains(manual_domain)
        if not routed_domains:
            routed = self.domain_router.route(inputs)
            routed_domains = self._ensure_domains(routed)

        if not routed_domains:
            fallback = self.domain_router.config.fallback_domains
            routed_domains = list(fallback[: self.domain_router.config.top_k])

        grouped_results: Dict[str, List[Document]] = {}
        for domain in routed_domains:
            req = self._with_domain_request(
                query=query,
                domain=domain,
                limit=self.staged_config.per_domain_limit,
            )
            grouped_results[domain] = self.knowledge_base.search(req)

        coarse_docs = self._coarse_rank(grouped_results, routed_domains)

        if self.colbert_reranker is not None and coarse_docs:
            final_docs = list(
                self.colbert_reranker.compress_documents(coarse_docs, query=query)
            )
        else:
            final_docs = coarse_docs[: self.staged_config.colbert.top_k]

        search_time = time.time() - t0
        return {
            "documents": final_docs,
            "search_time": search_time,
            "routed_domains": routed_domains,
            "candidate_size": len(coarse_docs),
        }
