"""
SearchGraph：单个查询的检索与合成子图

功能流程：
  查询输入 → [DB 检索] → [网络搜索(可选)] → [RAG 合成] → [质量评估(可选)] → 最终输出

特点：
- 支持本地数据库与网络两种检索源
- 集成反思机制：评估生成内容质量，判断是否需重试
- V2 网络搜索：支持参数化、结果筛选、URL 可访问性检测
"""

import html
import json
import logging
import re
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional

import httpx
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from MedicalRag.agent.tools import AgentTools
from MedicalRag.prompts.templates import get_prompt_template

from ..config.models import AppConfig
from ..core.ColBERTReranker import ColBERTReranker
from ..core.utils import create_llm_client, preload_embedding_clients
from .models import NetworkSearchResult, SearchMessagesState
from .tools import searxng_search

logger = logging.getLogger(__name__)


# ======================== 工具函数 ========================


def del_think(text: str) -> str:
    """移除 LLM 思考标签（如 <think>...</think>），返回实际内容"""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def json_to_list_document(text: str) -> list:
    """将 JSON 字符串转换为 Document 对象列表

    Args:
        text: JSON 字符串

    Returns:
        Document 列表，解析失败时返回空列表
    """
    if not text or not text.strip():
        logger.warning("工具返回内容为空，跳过文档解析")
        return []
    try:
        return [Document(**d) for d in json.loads(text)]
    except json.JSONDecodeError:
        logger.warning(f"工具返回内容非 JSON，跳过解析。内容片段：{text[:200]}")
        return []


def format_document_str(documents: List[Document]) -> str:
    """将文档列表格式化为可读的文本

    Args:
        documents: 文档列表

    Returns:
        格式化的文档文本（倒序取前 6 个）
    """
    parts = []
    for i, d in enumerate(reversed(documents)):
        if i >= 6:  # 最多显示 6 个文档
            break
        parts.append(f"## 文档{i + 1}：\n{d.page_content}\n")
    return "".join(parts)


# ======================== 格式化与处理函数 ========================


def _format_search_results(results: List[Dict[str, Any]], limit: int = 30) -> str:
    """格式化搜索结果为可读文本 拼成 prompt

    Args:
        results: 搜索结果列表
        limit: 最多显示条数

    Returns:
        格式化的文本
    """
    rows: List[str] = []
    for i, item in enumerate(results[:limit], start=1):
        title = item.get("title") or ""
        content = item.get("content") or item.get("snippet") or ""
        url = item.get("url") or item.get("link") or ""
        rows.append(f"[{i}] 标题: {title}\n摘要: {content}\nURL: {url}")
    return "\n\n".join(rows)


def _extract_numbers(text: str) -> List[int]:
    """从文本中提取所有整数"""
    return [int(x) for x in re.findall(r"\b\d+\b", text or "")]


def _pick_results(
    results: List[Dict[str, Any]], ids: List[int], top_k: int
) -> List[Dict[str, Any]]:
    """根据索引列表筛选结果

    Args:
        results: 候选结果列表
        ids: 要选取的索引（1-based）
        top_k: 最多返回数量

    Returns:
        筛选后的结果列表
    """
    picked: List[Dict[str, Any]] = []
    seen = set()
    for idx in ids:
        if idx in seen:
            continue
        seen.add(idx)
        pos = idx - 1
        if 0 <= pos < len(results):
            picked.append(results[pos])
        if len(picked) >= top_k:
            break
    return picked


def _strip_html_to_text(content: str) -> str:
    """将 HTML 内容转换为纯文本

    移除脚本、样式、标签等，解码 HTML 实体。

    Args:
        content: HTML 内容

    Returns:
        纯文本
    """
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", content)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _should_call_tool(last_ai: BaseMessage) -> bool:
    """判断上一步 LLM 消息是否触发了工具调用"""
    return bool(getattr(last_ai, "tool_calls", None))


def llm_db_search(
    state: SearchMessagesState,
    llm: BaseChatModel,
    db_tool_node: ToolNode,
    show_debug: bool,
) -> SearchMessagesState:
    """本地数据库检索节点

    调用 LLM 生成工具调用，检索本地医学知识库。

    Args:
        state: 执行状态
        llm: 绑定了 database_search 工具的 LLM
        db_tool_node: 工具执行节点
        show_debug: 是否打印调试日志

    Returns:
        更新后的状态（包含检索到的文档）
    """
    query = state["query"]
    call_db_tmpl = get_prompt_template("call_db")
    db_ai = llm.invoke(
        [
            SystemMessage(content=call_db_tmpl["system"]),
            HumanMessage(content=call_db_tmpl["user"].format(query=query)),
        ]
    )
    state["other_messages"].append(db_ai)
    if _should_call_tool(db_ai):
        if show_debug:
            logger.info(
                f"开始DB检索，检索参数：{db_ai.additional_kwargs['tool_calls'][0]['function']['arguments']}"
            )
        tool_msgs: ToolMessage = db_tool_node.invoke([db_ai])
        state["other_messages"].append(tool_msgs)
        state["docs"].extend(json_to_list_document(tool_msgs[0].content))
        if show_debug:
            if len(state["docs"]) >= 2:
                logger.info(f"本地检索完毕，获得 {len(state['docs'])} 条文档")
            else:
                logger.info(f"本地检索完毕，仅获得 1 条文档")
    return state


def llm_network_search(
    state: SearchMessagesState,
    judge_llm: BaseChatModel,
    network_search_llm: BaseChatModel,
    network_tool_node: ToolNode,
    network_search_v2_llm: Optional[BaseChatModel],
    network_tool_node_v2: Optional[ToolNode],
    agent_config,
    show_debug: bool,
) -> SearchMessagesState:
    """网络搜索节点

    智能判断是否需要网络搜索补充本地结果，支持两种模式：
    - V1：简单搜索与结果合并
    - V2：参数化搜索、结果筛选、URL 可访问性检测

    Args:
        state: 执行状态（包含本地检索结果）
        judge_llm: 判断是否需要搜索的 LLM
        network_search_llm: V1 网络搜索的 LLM（绑定工具）
        network_tool_node: V1 工具执行节点
        network_search_v2_llm: V2 网络搜索的 LLM（绑定工具）
        network_tool_node_v2: V2 工具执行节点
        agent_config: Agent 配置
        show_debug: 是否打印调试日志

    Returns:
        更新后的状态（包含补充的网络搜索文档）
    """
    if show_debug:
        logger.info("检查是否缺失资料需要网络搜索...")
    # 创建 Pydantic 解析器和容错解析器
    parser = PydanticOutputParser(pydantic_object=NetworkSearchResult)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=judge_llm)

    format_instructions = (
        parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
    )

    judge_messages = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                get_prompt_template("web_router")["system"].format(
                    format_instructions=format_instructions
                ),
            ),
            ("human", get_prompt_template("web_router")["user"]),
        ]
    )

    calling_messages = ChatPromptTemplate.from_messages(
        [
            ("system", get_prompt_template("call_web")["system"]),
            ("human", get_prompt_template("call_web")["user"]),
        ]
    )

    # 第 1 步：判断是否需要网络搜索
    judge_chain = (
        judge_messages
        | judge_llm
        | RunnableLambda(lambda x: del_think(x.content))
        | fixing_parser
    )

    try:
        result: NetworkSearchResult = judge_chain.invoke(
            {
                "query": state["query"],
                "docs": format_document_str(state.get("docs", [])),
            }
        )
        if show_debug:
            logger.info(
                f"网络搜索判断: {'需要' if result.need_search else '不需要'}，搜索词：{result.search_query}"
            )

        judge_ai_content = f"分析结果: {result.model_dump()}"
        judge_ai = AIMessage(content=judge_ai_content)
        state["other_messages"].append(judge_ai)

    except Exception as e:
        logger.error(f"网络搜索判断解析失败: {e}")
        result = NetworkSearchResult(
            need_search=False, search_query="", remain_doc_index=[]
        )
        judge_ai = AIMessage(content=f"解析失败，使用默认值不搜索")
        state["other_messages"].append(judge_ai)

    # 第 2 步：执行网络搜索（如需要）
    if result.need_search and result.search_query.strip():
        workflow = getattr(agent_config, "search_workflow_version", "v1")
        use_v2 = (
            workflow == "v2"
            and network_search_v2_llm is not None
            and network_tool_node_v2 is not None
        )

        if use_v2:
            # V2：参数化搜索 + 结果筛选 + 可访问性检测
            logger.info("使用 V2 网络搜索流程...")
            search_tmpl = get_prompt_template("search_tools")
            plan_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", search_tmpl["system"]),
                    ("human", search_tmpl["user"]),
                ]
            )
            search_query = result.search_query.strip()
            min_results = int(getattr(agent_config, "searx_v2_min_results", 20))
            max_results = int(getattr(agent_config, "searx_v2_max_results", 30))

            # V2.1：LLM 选择搜索参数并调用工具
            plan_ai = (plan_prompt | network_search_v2_llm).invoke(
                {
                    "query": search_query,
                    "min_results": min_results,
                    "max_results": max_results,
                    "current_date": datetime.now().isoformat(),
                }
            )
            state["other_messages"].append(plan_ai)

            raw_results: List[Dict[str, Any]] = []
            if _should_call_tool(plan_ai):
                tool_msgs: ToolMessage = network_tool_node_v2.invoke([plan_ai])
                state["other_messages"].append(tool_msgs)
                try:
                    raw_results = json.loads(tool_msgs[0].content or "[]")
                except Exception:
                    raw_results = []
            state["raw_web_results"] = raw_results

            if show_debug:
                logger.info(f"V2 网络检索返回 {len(raw_results)} 条候选结果")

            if raw_results:
                # V2.2：LLM 筛选 Top-K 结果
                top_k = int(getattr(agent_config, "searx_v2_select_top_k", 6))
                select_tmpl = get_prompt_template("select_best_result")
                select_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", select_tmpl["system"]),
                        ("human", select_tmpl["user"]),
                    ]
                )
                select_ai = (select_prompt | judge_llm).invoke(
                    {
                        "query": state["query"],
                        "context": _format_search_results(
                            raw_results, limit=max_results
                        ),
                        "top_k": top_k,
                    }
                )
                state["other_messages"].append(select_ai)
                ids = _extract_numbers(del_think(select_ai.content or ""))
                if not ids:
                    ids = list(range(1, min(top_k, len(raw_results)) + 1))

                picked = _pick_results(raw_results, ids=ids, top_k=top_k)
                state["selected_web_ids"] = ids[:top_k]
                state["selected_web_results"] = picked

                # V2.3：数据处理 - 检测 URL 可访问性、提取文本
                timeout = float(
                    getattr(agent_config, "searx_v2_fetch_timeout_sec", 5.0)
                )
                max_chars = int(
                    getattr(agent_config, "searx_v2_max_content_chars", 3000)
                )
                new_docs: List[Document] = []
                for item in picked:
                    title = item.get("title") or ""
                    snippet = item.get("content") or item.get("snippet") or ""
                    url = item.get("url") or item.get("link") or ""

                    page_text = snippet
                    accessible = False
                    if url:
                        try:
                            with httpx.Client(
                                timeout=timeout, follow_redirects=True
                            ) as client:
                                resp = client.get(url)
                                accessible = resp.status_code < 400
                                if accessible and resp.text:
                                    page_text = _strip_html_to_text(resp.text)[
                                        :max_chars
                                    ]
                        except Exception:
                            accessible = False

                    merged = "\n".join(p for p in [title, page_text, url] if p).strip()
                    if merged:
                        new_docs.append(
                            Document(
                                page_content=merged,
                                metadata={
                                    "source": "searxng",
                                    "title": title,
                                    "url": url,
                                    "accessible": accessible,
                                    "raw": item,
                                },
                            )
                        )

                # 合并文档：保留指定索引的本地文档 + 新增网络文档
                remain_doc = result.remain_doc_index
                if remain_doc:
                    valid_indices = [
                        i - 1 for i in remain_doc if 0 < i <= len(state.get("docs", []))
                    ]
                    state["docs"] = [state["docs"][i] for i in valid_indices]
                else:
                    state["docs"] = []

                state["docs"].extend(new_docs)
                if show_debug:
                    logger.info(f"V2 网络检索完毕，新增文档 {len(new_docs)} 条")
        else:
            # V1：简单搜索流程
            search_chain = calling_messages | network_search_llm
            search_ai = search_chain.invoke({"search_query": result.search_query})
            state["other_messages"].append(search_ai)
            if _should_call_tool(search_ai):
                tool_msgs: ToolMessage = network_tool_node.invoke([search_ai])
                state["other_messages"].append(tool_msgs)

                remain_doc = result.remain_doc_index
                if remain_doc:
                    valid_indices = [
                        i - 1 for i in remain_doc if 0 < i <= len(state.get("docs", []))
                    ]
                    state["docs"] = [state["docs"][i] for i in valid_indices]
                else:
                    state["docs"] = []

                state["docs"].extend(json_to_list_document(tool_msgs[0].content))
                if show_debug:
                    logger.info("V1 网络检索完毕")
    else:
        if show_debug:
            logger.info("资料已充分，无需网络搜索")

    return state


def rag(
    state: SearchMessagesState, llm: BaseChatModel, show_debug: bool
) -> SearchMessagesState:
    if show_debug:
        logger.info(f"开始RAG...")
    response_tmpl = get_prompt_template("response")
    sys = response_tmpl["system"]
    user = response_tmpl["user"]

    prompt = [
        SystemMessage(content=sys),
        HumanMessage(
            content=user.format(
                chat_history=state.get("chat_history", ""),
                all_document_str=format_document_str(state.get("docs", [])),
                input=state["query"],
            )
        ),
    ]

    rag_ai = llm.invoke(prompt)
    rag_ai.content = del_think(rag_ai.content)
    if not isinstance(state["main_messages"][-1], AIMessage):
        # 上一轮rag生成合格
        state["main_messages"].append(rag_ai)
    else:
        # 上一轮rag生成不合格，删除上一轮的信息
        state["main_messages"].pop()
        state["main_messages"].append(rag_ai)
    state["summary"] = rag_ai.content
    return state


def judge(
    state: SearchMessagesState, llm: BaseChatModel, show_debug: bool
) -> SearchMessagesState:
    """判断节点：负责判断和修改状态"""
    if show_debug:
        logger.info(f"开始评估...")
    judge_ai = llm.invoke(
        [
            SystemMessage(content=get_prompt_template("judge_rag")["system"]),
            HumanMessage(
                content=get_prompt_template("judge_rag")["user"].format(
                    format_document_str=format_document_str(state.get("docs", [])),
                    query=state["query"],
                    summary=state.get("summary", ""),
                )
            ),
        ]
    )
    result = del_think(judge_ai.content or "").strip().lower()
    if show_debug:
        logger.info(f"评估结果{result[:20]}")
    state["other_messages"].append(AIMessage(content=f"[JUDGE]={result}"))

    # 在这里修改状态
    if "y" in result:
        state["judge_result"] = "pass"
    else:
        retries_left = int(state.get("retry", 0))
        if retries_left > 0:
            state["retry"] = retries_left - 1  # 状态修改会被保存
            state["judge_result"] = "retry"
        else:
            state["judge_result"] = "fail"

    return state


class SearchGraph:
    def __init__(
        self,
        config: AppConfig,
        power_model: BaseChatModel,
        websearch_func=searxng_search,
    ) -> None:
        self.config = config
        preload_embedding_clients(self.config)
        self.show_debug = bool(getattr(self.config.agent, "console_debug", False))
        self.agent_tools = AgentTools(self.config)
        self.agent_tools.register_websearch(websearch_func)
        self.db_search_tool = self.agent_tools.make_database_search_tool()
        self.network_search_tool = self.agent_tools.make_web_search_tool()
        self.network_search_v2_tool = self.agent_tools.make_web_search_v2_tool()

        # bind_tools() 返回新的 RunnableBinding，不修改原模型，无需 deepcopy
        self.db_search_llm = power_model.bind_tools([self.db_search_tool])
        self.network_search_llm = power_model.bind_tools([self.network_search_tool])
        self.network_search_v2_llm = power_model.bind_tools(
            [self.network_search_v2_tool]
        )
        self.llm = create_llm_client(self.config.llm)

        # 真正执行 tool（函数调用）
        self.db_tool_node = ToolNode([self.db_search_tool])
        self.network_tool_node = ToolNode([self.network_search_tool])
        self.network_tool_node_v2 = ToolNode([self.network_search_v2_tool])
        self.search_graph = None

        # 可选：启动期预加载 ColBERT 模型，避免首次请求冷启动
        if (
            self.config.staged_retrieval.enabled
            and self.config.staged_retrieval.colbert.enabled
            and getattr(self.config.staged_retrieval.colbert, "preload", False)
        ):
            try:
                ColBERTReranker.preload(
                    model_name=self.config.staged_retrieval.colbert.model_name,
                    device=self.config.staged_retrieval.colbert.device,
                )
            except Exception as e:
                logger.warning("ColBERT 预加载失败，运行时再懒加载: %s", e)

    # ---------- 具有重试回路的图式构建 ----------
    def build_search_graph(self):
        """
        构建图
        """

        def judge_router(state: SearchMessagesState) -> str:
            """简单的路由函数：只读取状态，不修改"""
            return state.get("judge_result", "fail")

        def finish_success(state: SearchMessagesState) -> SearchMessagesState:
            """结束节点：成功输出"""
            state["final"] = (state.get("summary", "") or "").strip() or "（空）"
            return state

        def finish_fail(state: SearchMessagesState) -> SearchMessagesState:
            """结束节点：失败警告输出"""
            base = (state.get("summary", "") or "").strip() or "（空）"
            state["final"] = base + "\n\n（内容可能不属实）"
            return state

        g = StateGraph(SearchMessagesState)

        # 原子节点
        db_search_node = partial(
            llm_db_search,
            llm=self.db_search_llm,
            db_tool_node=self.db_tool_node,
            show_debug=self.show_debug,
        )
        g.add_node("db_search", db_search_node)
        network_search_node = partial(
            llm_network_search,
            judge_llm=self.llm,
            network_search_llm=self.network_search_llm,
            network_tool_node=self.network_tool_node,
            network_search_v2_llm=self.network_search_v2_llm,
            network_tool_node_v2=self.network_tool_node_v2,
            agent_config=self.config.agent,
            show_debug=self.show_debug,
        )
        g.add_node("web_search", network_search_node)
        rag_node = partial(rag, llm=self.llm, show_debug=self.show_debug)
        g.add_node("rag", rag_node)
        g.add_node("finish_success", finish_success)
        g.add_node("finish_fail", finish_fail)
        judge_node = partial(judge, llm=self.llm, show_debug=self.show_debug)
        g.add_node("judge", judge_node)
        # 入口
        g.set_entry_point("db_search")

        # db_search -> web_search
        if self.config.agent.network_search_enabled:
            g.add_edge("db_search", "web_search")
            g.add_edge("web_search", "rag")
        else:
            g.add_edge("db_search", "rag")

        # rag -> judge_router（条件分支）
        # judge -> 条件路由
        if self.config.agent.mode == "analysis":
            g.add_edge("rag", "judge")
            g.add_conditional_edges(
                "judge",  # 从判断节点出发
                judge_router,  # 纯路由函数
                {
                    "pass": "finish_success",
                    "retry": "rag",
                    "fail": "finish_fail",
                },
            )
            # 结束
            g.add_edge("finish_success", END)
            g.add_edge("finish_fail", END)
        elif self.config.agent.mode == "fast":
            g.add_edge("rag", END)

        self.search_graph = g.compile()

    # ---------- 对外：跑整张图，返回最终输出 ----------
    def answer(self, query: str) -> str:
        if self.search_graph is None:
            self.build_search_graph()
        init_state: SearchMessagesState = {
            "query": query,
            "chat_history": "",
            "main_messages": [HumanMessage(content=query)],
            "other_messages": [],
            "docs": [],
            "raw_web_results": [],
            "selected_web_ids": [],
            "selected_web_results": [],
            "summary": "",
            "retry": self.config.agent.max_attempts,
            "final": "",
        }
        # 执行图
        out_state: SearchMessagesState = self.search_graph.invoke(init_state)
        return out_state.get("final", "") or out_state.get("summary", "") or "（空）"

    def run(self, init_state: SearchMessagesState) -> SearchMessagesState:
        if self.search_graph is None:
            self.build_search_graph()
        out_state: SearchMessagesState = self.search_graph.invoke(init_state)
        return out_state
