from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from langchain_core.documents import Document

from MedicalRag.config.loader import ConfigLoader
from MedicalRag.rag.SimpleRag import SimpleRAG


@dataclass(frozen=True)
class BasicRagOptions:
    domain: Optional[str] = None
    routing_method: Optional[Literal["centroid", "llm"]] = None
    routing_top_k: Optional[int] = None


@dataclass(frozen=True)
class BasicRagResult:
    answer: str
    documents: list[Document]
    search_time: float
    generation_time: float
    routed_domains: list[str]
    candidate_size: Optional[int]
    selected_domain: Optional[str]


class BasicRagRunner:
    def __init__(self, config_path: Optional[str] = None):
        self.config_loader = ConfigLoader(config_path)
        self.rag = SimpleRAG(self.config_loader.config)

    def run(
        self, query: str, options: Optional[BasicRagOptions] = None
    ) -> BasicRagResult:
        opts = options or BasicRagOptions()
        raw = self.rag.rag_chain.invoke(
            {
                "input": query,
                "domain": opts.domain,
                "routing_method": opts.routing_method,
                "routing_top_k": opts.routing_top_k,
            }
        )

        raw_dict = raw if isinstance(raw, dict) else {}
        milvus_result = raw_dict.get("milvus_result", {})
        llm_result = raw_dict.get("llm", {})

        milvus_result_dict = milvus_result if isinstance(milvus_result, dict) else {}
        llm_result_dict = llm_result if isinstance(llm_result, dict) else {}

        documents = milvus_result_dict.get("documents", [])
        routed_domains = milvus_result_dict.get("routed_domains", [])
        candidate_size = milvus_result_dict.get("candidate_size")

        normalized_documents = [d for d in documents if isinstance(d, Document)]
        normalized_routed_domains = [d for d in routed_domains if isinstance(d, str)]

        return BasicRagResult(
            answer=str(llm_result_dict.get("answer", "")),
            documents=normalized_documents,
            search_time=_to_float(milvus_result_dict.get("search_time")),
            generation_time=_to_float(llm_result_dict.get("generate_time")),
            routed_domains=normalized_routed_domains,
            candidate_size=_to_optional_int(candidate_size),
            selected_domain=_normalize_domain(raw_dict.get("domain")),
        )


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_optional_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_domain(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    return None


def build_basic_rag_report(
    result: BasicRagResult,
    *,
    show_docs: bool = False,
    show_index: bool = False,
    docs_preview_count: int = 3,
    score_preview_count: int = 10,
    content_preview_chars: int = 200,
) -> str:
    lines: list[str] = []
    lines.append(
        f"\n检索完成，检索用时：{result.search_time} s，生成用时：{result.generation_time} s\n"
    )
    lines.append(result.answer)

    if show_index:
        lines.append("\n=== RAG 索引信息 ===")
        routed = (
            result.routed_domains if result.routed_domains else result.selected_domain
        )
        lines.append(f"路由结果(top-k): {routed}")
        if result.candidate_size is not None:
            lines.append(f"merge后候选数: {result.candidate_size}")
        lines.append(f"最终返回文档数: {len(result.documents)}")

        if result.documents:
            lines.append("\nTop文档打分（前10条）:")
            for i, doc in enumerate(result.documents[:score_preview_count], 1):
                metadata = doc.metadata or {}
                lines.append(
                    f"{i}. domain={metadata.get('domain')} "
                    f"distance={metadata.get('distance')} "
                    f"retrieval_rank={metadata.get('retrieval_rank')} "
                    f"routed_domain_rank={metadata.get('routed_domain_rank')} "
                    f"coarse_score={metadata.get('coarse_score')} "
                    f"colbert_score={metadata.get('colbert_score')}"
                )

    if show_docs:
        lines.append(
            f"\n参考资料 ({len(result.documents)} 条)，展示前{docs_preview_count}条:\n"
        )
        for i, doc in enumerate(result.documents[:docs_preview_count], 1):
            metadata = doc.metadata or {}
            lines.append(
                f"{i}. 数据源：{metadata.get('source')} "
                f"数据源名：{metadata.get('source_name')} "
                f"领域：{metadata.get('domain')} "
                f"向量距离：{metadata.get('distance')}\n"
            )
            content = (
                doc.page_content[:content_preview_chars] + "..."
                if len(doc.page_content) > content_preview_chars
                else doc.page_content
            )
            lines.append(f"{content}\n")

    return "\n".join(lines)
