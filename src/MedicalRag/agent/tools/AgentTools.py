"""
Agent 工具集

提供 LLM 可调用的工具：网络搜索（v1/v2）、本地数据库检索、计算器等
"""

from typing import List, Dict, Optional
import ast
import operator as op
import json
import logging

from langchain.tools import tool
from langchain_core.documents import Document

from ...config.models import AppConfig, SearchRequest
from ...core.KnowledgeBase import MedicalHybridKnowledgeBase
from ...core.HybridRetriever import MedicalHybridRetriever
from ...core.StagedHybridRetriever import StagedHybridRetriever
from ...rag.DomainRouter import DomainRouter
from ...core.utils import create_llm_client
from ...core.DBFactory import get_kb

logger = logging.getLogger(__name__)


class AgentTools:
    """LLM Agent 工具集合"""

    def __init__(self, app_config: AppConfig) -> None:
        """初始化工具集

        Args:
            app_config: 应用配置对象
        """
        self.app_config = app_config
        self.WEBSEARCH_FUNC = None

    def register_websearch(self, func):
        """注册网络搜索函数

        Args:
            func: 网络搜索实现函数
        """
        self.WEBSEARCH_FUNC = func

    def make_web_search_tool(self):
        """创建网络搜索工具（V1 版本：简单调用）

        Returns:
            tool: LLM 可调用的网络搜索工具
        """
        if self.WEBSEARCH_FUNC is None:
            raise ValueError("未注册网络检索工具，请先调用 register_websearch()")

        cnt = self.app_config.agent.network_search_cnt
        agent_cfg = self.app_config.agent
        web_kwargs = {
            "searx_host": agent_cfg.searx_host,
            "language": agent_cfg.searx_language,
            "engines": agent_cfg.searx_engines,
            "categories": agent_cfg.searx_categories,
            "time_range": agent_cfg.searx_time_range,
            "safesearch": agent_cfg.searx_safe_search,
        }

        @tool("web_search")
        def web_search(query: str) -> str:
            """使用输入文本进行联网搜索，返回 JSON 格式的搜索结果"""
            try:
                results: List[Document] = self.WEBSEARCH_FUNC(
                    query=query, cnt=cnt, **web_kwargs
                )
            except TypeError:
                # 兼容旧函数签名：func(query, cnt)
                results = self.WEBSEARCH_FUNC(query, cnt)
            return json.dumps([d.model_dump() for d in results], ensure_ascii=False)

        return web_search

    def make_web_search_v2_tool(self):
        """创建网络搜索工具（V2 版本：支持参数自定义）

        V2 版本允许 LLM 动态选择搜索参数（结果数、语言、时间范围等），
        更灵活地适应不同查询场景。

        Returns:
            tool: LLM 可调用的网络搜索工具
        """
        if self.WEBSEARCH_FUNC is None:
            raise ValueError("未注册网络检索工具，请先调用 register_websearch()")

        agent_cfg = self.app_config.agent
        default_language = agent_cfg.searx_language
        default_engines = agent_cfg.searx_engines
        default_categories = agent_cfg.searx_categories
        default_time_range = agent_cfg.searx_time_range
        default_safesearch = agent_cfg.searx_safe_search
        min_results = int(agent_cfg.searx_v2_min_results)
        max_results = int(agent_cfg.searx_v2_max_results)

        @tool("web_search_v2")
        def web_search_v2(
            query: str,
            num_results: int,
            language: Optional[str] = None,
            safesearch: Optional[int] = None,
            time_range: Optional[str] = None,
            categories: Optional[List[str]] = None,
        ) -> str:
            """参数化网络搜索：允许 LLM 选择搜索参数

            Args:
                query: 搜索查询词
                num_results: 结果数量
                language: 搜索语言（默认使用配置值）
                safesearch: 安全搜索级别 0-2
                time_range: 时间范围过滤
                categories: 搜索类别

            Returns:
                JSON 格式的原始搜索结果
            """
            try:
                # 结果数在最小值和最大值之间
                k = max(min_results, min(int(num_results), max_results))
                results = self.WEBSEARCH_FUNC(
                    query=query,
                    cnt=k,
                    searx_host=agent_cfg.searx_host,
                    language=language or default_language,
                    engines=default_engines,
                    categories=categories or default_categories,
                    time_range=time_range
                    if time_range is not None
                    else default_time_range,
                    safesearch=default_safesearch if safesearch is None else safesearch,
                    return_raw=True,
                )
            except TypeError:
                # 兼容旧签名函数
                docs = self.WEBSEARCH_FUNC(query, max_results)
                results = [d.metadata.get("raw", d.metadata) for d in docs]
            return json.dumps(results, ensure_ascii=False)

        return web_search_v2

    def make_database_search_tool(self):
        """创建本地数据库检索工具

        支持两种检索模式：
        1. 分阶段检索（Staged）：先用领域路由筛选，再进行混合检索
        2. 普通混合检索（Hybrid）：直接进行向量+关键词混合检索

        Returns:
            tool: LLM 可调用的数据库检索工具
        """

        @tool("database_search")
        def database_search(search_config: SearchRequest) -> str:
            """在本地医学知识库中进行检索

            Args:
                search_config: 检索配置（包含查询、领域、检索参数等）

            Returns:
                JSON 格式的检索文档列表
            """
            search_req = (
                search_config
                if isinstance(search_config, SearchRequest)
                else SearchRequest.model_validate(search_config)
            )
            kb = get_kb(self.app_config.model_dump())

            # 选择检索模式
            if self.app_config.staged_retrieval.enabled:
                # 分阶段检索：使用领域路由进行智能筛选
                routing_llm = create_llm_client(self.app_config.llm)
                domain_router = DomainRouter(
                    embedding=kb.summary_embedding,
                    default_llm=routing_llm,
                    routing_config=self.app_config.routing,
                    app_config=self.app_config,
                )
                retriever = StagedHybridRetriever(
                    knowledge_base=kb,
                    search_config=search_req,
                    domain_router=domain_router,
                    staged_config=self.app_config.staged_retrieval,
                )
                retrieved = retriever.invoke(
                    {"input": search_req.query, "domain": search_req.domain}
                )
                results = retrieved.get("documents", [])
            else:
                # 普通混合检索
                retriever = MedicalHybridRetriever(kb, search_req)
                retrieved = retriever.invoke(
                    {"input": search_req.query, "domain": search_req.domain}
                )
                results = retrieved.get("documents", [])

            return json.dumps([d.model_dump() for d in results], ensure_ascii=False)

        return database_search

    def make_calculator_tool(self):
        """创建计算器工具

        支持基础算术运算：加、减、乘、除、负号

        Returns:
            tool: LLM 可调用的计算器工具
        """
        # 定义支持的运算符
        operators = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.USub: op.neg,
        }

        def eval_expr(expr: str) -> float:
            """安全地计算表达式（防注入）

            Args:
                expr: 算术表达式字符串

            Returns:
                计算结果

            Raises:
                ValueError: 表达式包含不支持的操作
            """

            def _eval(node):
                if isinstance(node, ast.Num):  # 数值常量
                    return node.n
                elif isinstance(node, ast.BinOp):  # 二元运算（如 a + b）
                    return operators[type(node.op)](_eval(node.left), _eval(node.right))
                elif isinstance(node, ast.UnaryOp):  # 一元运算（如 -a）
                    return operators[type(node.op)](_eval(node.operand))
                else:
                    raise ValueError("不支持的表达式类型")

            node = ast.parse(expr, mode="eval").body
            return _eval(node)

        @tool("calculator")
        def calculator(expression: str) -> str:
            """计算算术表达式

            支持的操作：加(+)、减(-)、乘(*)、除(/)、负号(-)

            Args:
                expression: 算术表达式字符串，如 "2 + 3 * 4"

            Returns:
                计算结果或错误信息
            """
            try:
                result = eval_expr(expression)
                return str(result)
            except Exception as e:
                return f"计算出错: {e}"

        return calculator
