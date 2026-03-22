from __future__ import annotations

import logging
import re
import traceback
from typing import List, Optional, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from ..config.models import AppConfig, SearchRequest
from ..core.HybridRetriever import MedicalHybridRetriever
from ..core.StagedHybridRetriever import StagedHybridRetriever
from ..core.KnowledgeBase import MedicalHybridKnowledgeBase
from ..core.utils import create_llm_client
from ..prompts.templates import get_prompt_template
from .RagBase import BasicRAG
from .DomainRouter import DomainRouter

logger = logging.getLogger(__name__)


class SimpleRAG(BasicRAG):
    """基础医疗RAG系统（工程化路由版）"""

    def __init__(
        self, config: AppConfig, search_config: Optional[SearchRequest] = None
    ):
        super().__init__(config, search_config)
        self.knowledge_base = MedicalHybridKnowledgeBase(config)

        self.llm = create_llm_client(config.llm)
        self.prompt = self._setup_dialogue_rag_prompt()

        # 路由器：由 config.routing 统一控制（method / llm / centroid_file）
        self.domain_router = DomainRouter(
            embedding=self.knowledge_base.summary_embedding,
            default_llm=self.llm,
            routing_config=config.routing,
            app_config=config,
        )

        self.milvus_retriever: BaseRetriever = self._build_retriever()

        self._setup_chain()
        logger.info("SimpleRAG 系统初始化完成")

    def _build_retriever(self) -> BaseRetriever:
        if self.config.staged_retrieval.enabled:
            logger.info("启用分阶段检索器（含ColBERT重排）")
            return StagedHybridRetriever(
                self.knowledge_base,
                self.search_config,
                self.domain_router,
                self.config.staged_retrieval,
            )

        logger.info("启用基础混合检索器")
        return MedicalHybridRetriever(self.knowledge_base, self.search_config)

    def _setup_dialogue_rag_prompt(self) -> ChatPromptTemplate:
        template = get_prompt_template("basic_rag")
        if isinstance(template, dict):
            return ChatPromptTemplate.from_messages(
                [("system", template["system"]), ("human", template["user"])]
            )
        return ChatPromptTemplate.from_template(template)

    def _setup_chain(self):
        def format_document_str(inputs: dict) -> str:
            documents: List[Document] = inputs["milvus_result"]["documents"]
            return "".join(
                [
                    f"## 文档{i + 1}：\n{d.page_content}\n"
                    for i, d in enumerate(documents)
                ]
            )

        def strip_think_and_time(msg: AIMessage):
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
            dur = msg.response_metadata.get("total_duration", 0) / 1e9
            return {"answer": cleaned.strip(), "generate_time": dur}

        # 0) 路由（RunnableLambda）：把 domain 注入上下文
        route = RunnablePassthrough.assign(
            domain=self.domain_router.as_runnable()
        ).with_config(run_name="route")

        # 1) 检索
        retrieve = RunnablePassthrough.assign(
            milvus_result=self.milvus_retriever
        ).with_config(run_name="retrieve_documents")

        # 2) 文档拼接
        format_docs = RunnablePassthrough.assign(
            all_document_str=RunnableLambda(format_document_str)
        ).with_config(run_name="format_doc")

        # 3) 生成
        generate = (
            self.prompt.with_config(run_name="apply_prompt")
            | self.llm.with_config(run_name="generate")
            | RunnableLambda(strip_think_and_time)
        )

        self.rag_chain = (
            route | retrieve | format_docs | RunnablePassthrough.assign(llm=generate)
        ).with_config(run_name="rag")
        logger.info("RAG链构建完成")

    def answer(
        self,
        query: str,
        return_document: bool = False,
        domain: Optional[str] = None,
        routing_method: Optional[Literal["centroid", "llm"]] = None,
        routing_top_k: Optional[int] = None,
    ) -> dict:
        effective_routing_method = routing_method or self.config.routing.method
        effective_routing_top_k = routing_top_k or self.config.routing.top_k
        logger.info(
            "处理问题: %s, domain: %s, routing_method: %s, routing_top_k: %s",
            query,
            domain,
            effective_routing_method,
            effective_routing_top_k,
        )
        try:
            result = self.rag_chain.invoke(
                {
                    "input": query,
                    "domain": domain,
                    # 允许请求级覆盖；为空时 DomainRouter 使用 config.routing.method
                    "routing_method": routing_method,
                    "routing_top_k": routing_top_k,
                }
            )
            answer = result["llm"]["answer"]
            if return_document:
                return {
                    "answer": answer,
                    "documents": result["milvus_result"]["documents"],
                    "search_time": result["milvus_result"]["search_time"],
                    "generation_time": result["llm"]["generate_time"],
                }
            return {
                "answer": answer,
                "search_time": result["milvus_result"]["search_time"],
                "generation_time": result["llm"]["generate_time"],
            }
        except Exception as e:
            logger.error(f"RAG处理失败: {e}")
            print(traceback.format_exc())
            error_msg = "抱歉，处理您的问题时出现错误，请稍后再试。"
            if return_document:
                return {
                    "answer": error_msg,
                    "documents": [],
                    "search_time": 0,
                    "generation_time": 0,
                }
            return {"answer": error_msg, "search_time": 0, "generation_time": 0}

    def update_search_config(self, search_config: SearchRequest):
        self.search_config = search_config
        self.milvus_retriever = self._build_retriever()
        self._setup_chain()
        logger.info(f"搜索配置已更新: {search_config}")
