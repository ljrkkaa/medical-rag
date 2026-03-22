"""
Agent 相关的数据模型与状态定义

集中管理 Pydantic 模型与 TypedDict，降低模块间耦合。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field


class AskMess(BaseModel):
    """追问判断的输出格式"""

    need_ask: bool = Field(default=False, description="是否需要向用户追问")
    questions: List[str] = Field(default_factory=list, description="追问的问题列表")


class SplitQuery(BaseModel):
    """查询拆分与改写的输出格式"""

    need_split: bool = Field(default=False, description="是否需要拆分成多个子查询")
    sub_query: List[str] = Field(
        default_factory=list,
        description="子查询列表（最多 3 个，每个子查询应相对独立）",
    )
    rewrite_query: str = Field(
        default="", description="不拆分时的改写查询（更便于检索）"
    )


class SearchMessagesState(TypedDict, total=False):
    """单查询执行的状态"""

    query: str
    chat_history: str
    main_messages: List[Union[HumanMessage, AIMessage]]
    other_messages: List[BaseMessage]       #  tool / judge / debug信息
    docs: List[Document]
    raw_web_results: List[Dict[str, Any]]
    selected_web_ids: List[int]
    selected_web_results: List[Dict[str, Any]]
    summary: str
    retry: int
    final: str
    judge_result: str


class NetworkSearchResult(BaseModel):
    """网络搜索判断结果"""

    need_search: bool = Field(description="是否需要进行网络搜索")
    search_query: str = Field(description="网络搜索查询词", default="")
    remain_doc_index: List[int] = Field(
        description="保留的本地文档索引列表", default=[]
    )


class SearxSearchArgs(BaseModel):
    """Searx 搜索参数"""

    query: str
    num_results: int = Field(default=30, ge=1, le=100)
    language: Optional[str] = "zh-CN"
    safesearch: Optional[int] = Field(default=1, ge=0, le=2)
    time_range: Optional[str] = None
    categories: Optional[List[str]] = None


class SelectResultIds(BaseModel):
    """结果筛选的索引选择"""

    ids: List[int] = Field(default_factory=list)
