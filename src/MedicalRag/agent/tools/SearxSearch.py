from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import logging

from langchain_community.utilities import SearxSearchWrapper
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def searxng_search(
    query: str,
    cnt: int,
    searx_host: str = "http://127.0.0.1:8081",
    language: Optional[str] = None,
    engines: Optional[List[str]] = None,
    categories: Optional[Union[str, List[str]]] = None,
    time_range: Optional[str] = None,
    safesearch: Optional[int] = None,
    return_raw: bool = False,
) -> Union[List[Document], List[Dict[str, Any]]]:
    """使用 SearxNG 进行网络检索。默认返回 Document 列表，V2 可返回原始结果。"""
    try:
        wrapper = SearxSearchWrapper(searx_host=searx_host, k=max(1, int(cnt)))

        kwargs: Dict[str, Any] = {"num_results": max(1, int(cnt))}
        if language:
            kwargs["language"] = language
        if engines:
            kwargs["engines"] = engines
        if categories:
            kwargs["categories"] = categories
        if time_range:
            kwargs["time_range"] = time_range
        if safesearch is not None:
            kwargs["safesearch"] = safesearch

        results = wrapper.results(query, **kwargs)
        if return_raw:
            return results[: max(1, int(cnt))]

        docs: List[Document] = []
        for item in results:
            title = item.get("title", "")
            snippet = item.get("snippet", "") or item.get("content", "")
            url = item.get("link", "") or item.get("url", "")

            page_content = "\n".join(p for p in [title, snippet, url] if p).strip()
            if not page_content:
                page_content = str(item)

            metadata = {
                "source": "searxng",
                "title": title,
                "url": url,
                "engine": item.get("engine"),
                "category": item.get("category"),
                "score": item.get("score"),
                "publishedDate": item.get("publishedDate"),
                "raw": item,
            }
            docs.append(Document(page_content=page_content, metadata=metadata))

        return docs[: max(1, int(cnt))]
    except Exception as exc:
        logger.warning("SearxNG 检索失败: %s", exc)
        return []
