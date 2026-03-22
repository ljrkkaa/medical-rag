"""
MedicalAgent：医学多轮对话 Agent

主流程：
  用户输入 → [ask] 判断是否需要追问
    ├─ 需要追问 → 打印问题，等待用户回复 → 继续下一轮
    └─ 信息充分 → [extract] → [split_query] → [run_query（并行）] → [answer（LLM合成）]

设计特点：
- 智能追问：自动识别信息缺陷，提出精准问题
- 多轮压缩：使用 running_summary 压缩历史，避免 Token 爆炸
- 并行检索：通过 LangGraph Send API 并行运行多个子查询
- 灵活重试：支持检索失败重试机制
"""

from __future__ import annotations

import logging
import re
from functools import partial
from operator import add
from typing import Annotated, Any, List, TypedDict

from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from ..config.models import AppConfig
from ..core.utils import create_llm_client
from .SearchGraph import SearchGraph
from .models import AskMess, SplitQuery, SearchMessagesState
from .utils import strip_think_get_tokens
from MedicalRag.prompts.templates import get_prompt_template

logger = logging.getLogger(__name__)


# ======================== 输出数据模型 ========================


def _history_to_text(messages: List[BaseMessage]) -> str:
    """将消息列表转换为可读的文本格式

    Args:
        messages: 消息列表

    Returns:
        格式化的对话文本
    """
    rows: List[str] = []
    for m in messages:
        role = "用户" if isinstance(m, HumanMessage) else "助手"
        rows.append(f"{role}: {m.content}")
    return "\n".join(rows).strip()


# ======================== 主状态定义 ========================


class MedicalAgentState(TypedDict, total=False):
    """MedicalAgent 的完整状态定义

    包含以下几类：
    1. 对话历史与用户信息
    2. 检索与回答过程
    3. 中间变量
    4. 性能监控
    """

    # 对话与用户画像
    dialogue_messages: List[BaseMessage]  # 完整对话历史（用户/助手）
    asking_messages: List[List[BaseMessage]]  # 追问阶段的消息（二维数组）
    background_info: str  # 用户背景信息摘要
    ask_obj: AskMess  # 追问判断结果
    multi_summary: List[str]  # 多轮摘要列表（最近 8-10 条）
    running_summary: str  # 压缩后的历史摘要（>8条时压缩）
    curr_input: str  # 当前用户输入

    # 检索与问题处理
    sub_query: SplitQuery  # 查询拆分结果
    rewritten_query: str  # 最终发送给检索的查询
    # 使用 Annotated[..., add] 进行结果聚合：
    # 多个 search_one 节点的返回值自动通过 add 合并到列表
    sub_query_results: Annotated[List[SearchMessagesState], add]

    # 流程控制
    max_ask_num: int  # 最大追问次数
    curr_ask_num: int  # 当前追问次数

    # 输出
    final_answer: str  # 最终答案
    performance: List[Any]  # 性能监控数据
    show_debug: bool  # 调试日志开关


# ======================== 图节点函数 ========================


def ask_judge(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """追问判断节点

    判断当前信息是否充分，如不充分则生成追问问题。
    支持多轮追问，最多可追问 max_ask_num 次。

    Args:
        state: Agent 状态
        llm: LLM 模型

    Returns:
        更新后的状态（包含追问判断结果）
    """
    parser = PydanticOutputParser(pydantic_object=AskMess)
    fixing = OutputFixingParser.from_llm(parser=parser, llm=llm)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                get_prompt_template("ask_user")["system"].format(
                    format_instructions=parser.get_format_instructions()
                    .replace("{", "{{")
                    .replace("}", "}}")
                ),
            ),
            MessagesPlaceholder(variable_name="asking_history"),
            ("human", get_prompt_template("ask_user")["user"]),
        ]
    )

    # 获取本轮追问历史
    curr_ask_mess = [] if state["curr_ask_num"] == 0 else state["asking_messages"][-1]
    ai = (prompt | llm | RunnableLambda(strip_think_get_tokens)).invoke(
        {
            "background_info": state["background_info"],
            "question": state["curr_input"],
            "asking_history": curr_ask_mess,
        }
    )

    # 更新追问消息记录
    if state["curr_ask_num"] == 0:
        state["asking_messages"].append([HumanMessage(content=state["curr_input"])])
    else:
        state["asking_messages"][-1].append(HumanMessage(content=state["curr_input"]))

    patch: AskMess = fixing.parse(ai["msg"])
    state["ask_obj"] = patch

    if patch.need_ask:
        state["asking_messages"][-1].append(
            AIMessage(content="\n".join(patch.questions))
        )
    else:
        state["asking_messages"][-1].append(AIMessage(content="信息已充分，可以回答"))

    state["performance"].append(("ask", ai))
    state["curr_ask_num"] += 1
    return state


def route_ask_again(state: MedicalAgentState) -> str:
    """追问路由

    决定是否继续追问或进入后续处理：
    - "ask"  → 需要追问，结束本轮，等待用户下一条消息
    - "pass" → 信息充分（或已达最大追问次数），继续后续处理
    """
    ask_obj = state.get("ask_obj")
    if ask_obj is None:
        return "pass"
    if ask_obj.need_ask and state["curr_ask_num"] < state["max_ask_num"]:
        return "ask"
    return "pass"


def extract_background_info(
    state: MedicalAgentState, llm: BaseChatModel
) -> MedicalAgentState:
    """信息抽取节点

    从追问阶段的多轮对话中抽取用户的关键背景信息
    （如年龄、症状、病史等），为后续检索提供上下文。
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_prompt_template("extract_user_info")["system"]),
            MessagesPlaceholder(variable_name="asking_history"),
            ("human", get_prompt_template("extract_user_info")["user"]),
        ]
    )
    asking_hist = state["asking_messages"][-1] if state["asking_messages"] else []
    ai = (prompt | llm | RunnableLambda(strip_think_get_tokens)).invoke(
        {
            "question": asking_hist[0].content if asking_hist else state["curr_input"],
            "asking_history": asking_hist,
        }
    )
    state["performance"].append(("extract", ai))
    state["background_info"] = ai["msg"]
    return state


def check_update_background(
    state: MedicalAgentState, llm: BaseChatModel
) -> MedicalAgentState:
    """背景更新节点

    第二轮及以后：检查用户新输入是否包含纠正或补充背景信息。
    如有则进行更新，避免过时的用户信息影响检索。
    """
    tmpl = get_prompt_template("update_background")
    result = llm.invoke(
        [
            SystemMessage(content=tmpl["system"]),
            HumanMessage(
                content=tmpl["user"].format(
                    background_info=state.get("background_info", ""),
                    question=state["curr_input"],
                )
            ),
        ]
    )
    updated = re.sub(
        r"<think>.*?</think>\s*", "", result.content, flags=re.DOTALL
    ).strip()
    state["background_info"] = updated
    return state


def condense_question(
    state: MedicalAgentState, llm: BaseChatModel
) -> MedicalAgentState:
    """问题改写节点（Condense Question Chain）

    当存在对话历史时，将当前问题改写成相对独立的查询，
    使其可以脱离历史上下文单独检索，避免检索器误解。
    """
    if not state.get("dialogue_messages"):
        # 首轮直接使用原始问题
        state["rewritten_query"] = state["curr_input"]
        return state

    tmpl = get_prompt_template("rephrase")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", tmpl["system"]),
            ("human", tmpl["user"]),
        ]
    )
    ai = (prompt | llm | RunnableLambda(strip_think_get_tokens)).invoke(
        {
            "chat_history": _history_to_text(state.get("dialogue_messages", [])),
            "question": state["curr_input"],
        }
    )
    state["rewritten_query"] = ai["msg"].strip() or state["curr_input"]
    state["performance"].append(("condense", ai))
    return state


def route_entry(state: MedicalAgentState) -> str:
    """入口路由

    决定是否进入追问流程：
    - 无背景信息 → 进入追问流程
    - 有背景信息 → 跳过追问，直接更新背景并检索
    """
    return "check_update_background" if state.get("background_info") else "ask"


def judge_split_query(
    state: MedicalAgentState, llm: BaseChatModel
) -> MedicalAgentState:
    """查询拆分节点

    判断是否需要将查询拆分成多个独立的子查询，
    以便并行检索多个角度的信息。
    同时在信息不需要拆分时，提供改写后的查询。
    """
    parser = PydanticOutputParser(pydantic_object=SplitQuery)
    fixing = OutputFixingParser.from_llm(parser=parser, llm=llm)

    # 组织历史摘要（包括压缩摘要和最近摘要）
    running = state.get("running_summary", "")
    recent = "\n".join(state.get("multi_summary", []))
    summary_context = (running + "\n" + recent).strip() if running else recent

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                get_prompt_template("handle_query")["system"].format(
                    format_instructions=parser.get_format_instructions()
                    .replace("{", "{{")
                    .replace("}", "}}"),
                    summary=summary_context,
                ),
            ),
            MessagesPlaceholder(variable_name="dialogue_messages"),
            ("user", get_prompt_template("handle_query")["user"]),
        ]
    )
    ai = (prompt | llm | RunnableLambda(strip_think_get_tokens)).invoke(
        {
            "background_info": state["background_info"],
            "question": (state.get("rewritten_query") or state["curr_input"]),
            "dialogue_messages": state["dialogue_messages"],
        }
    )
    patch: SplitQuery = fixing.parse(ai["msg"])
    # 限制最多 3 个子查询
    if len(patch.sub_query) > 3:
        patch.sub_query = patch.sub_query[:3]

    state["sub_query"] = patch
    state["performance"].append(("split_query", ai))
    return state


def route_to_subgraphs(state: MedicalAgentState) -> List[Send]:
    """并行分发路由

    使用 LangGraph 的 Send API 将查询分发到多个 search_one 节点并行执行。
    各子查询的返回结果通过 add reducer 自动合并到 sub_query_results 列表。
    """
    sq: SplitQuery = state.get("sub_query")
    queries: List[str] = []

    if sq and sq.need_split and sq.sub_query:
        queries = sq.sub_query[:3]
    else:
        base_q = (sq.rewrite_query if sq and sq.rewrite_query else "").strip()
        queries = [base_q or state["curr_input"]]

    if state.get("show_debug", False):
        logger.info(f"[路由] 拆分为 {len(queries)} 个子查询: {queries}")
    chat_history = _history_to_text(state.get("dialogue_messages", []))
    return [
        Send("search_one", {"query": q, "chat_history": chat_history}) for q in queries
    ]


def search_one(task_input: dict, search_graph: SearchGraph) -> dict:
    """单个子查询执行节点

    每个 Send 实例会以独立任务运行该节点。
    返回值通过 sub_query_results 的 add reducer 自动合并。

    Args:
        task_input: 包含 query 和 chat_history
        search_graph: SearchGraph 实例

    Returns:
        包含 sub_query_results 的字典，值为包含单个结果的列表
    """
    query = task_input["query"]
    init_state: SearchMessagesState = {
        "query": query,
        "chat_history": task_input.get("chat_history", ""),
        "main_messages": [HumanMessage(content=query)],
        "other_messages": [],
        "docs": [],
        "raw_web_results": [],
        "selected_web_ids": [],
        "selected_web_results": [],
        "summary": "",
        "retry": search_graph.config.agent.max_attempts,
        "final": "",
    }
    result = search_graph.run(init_state)
    return {"sub_query_results": [result]}


def gather_answer(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """答案聚合节点

    汇总各子查询的答案：
    - 单子查询：直接使用检索结果
    - 多子查询：用 LLM 综合多份子答案为统一回复

    同时更新：
    - dialogue_messages：加入当前问答
    - multi_summary：记录本轮摘要
    - running_summary：当摘要数过多时进行压缩
    """
    state["curr_ask_num"] = 0
    sub_results: List[SearchMessagesState] = state.get("sub_query_results", [])

    # 根据子查询结果数量，选择不同的答案生成策略
    if not sub_results:
        final_answer = "抱歉，检索未能获取到相关资料，请稍后再试。"

    elif len(sub_results) == 1:
        # 单个子查询：直接使用其结果
        final_answer = (
            sub_results[0].get("final") or sub_results[0].get("summary") or ""
        ).strip()
        if not final_answer:
            final_answer = "抱歉，根据提供的资料无法回答您的问题。"

    else:
        # 多个子查询：用 LLM 综合各子答案
        sub_answers = []
        for i, res in enumerate(sub_results):
            ans = (res.get("final") or res.get("summary") or "").strip()
            if ans:
                sub_answers.append(f"### 子问题 {i + 1} 分析：\n{ans}")

        if not sub_answers:
            final_answer = "抱歉，根据提供的资料无法回答您的问题。"
        else:
            # 准备综合上下文
            background = state.get("background_info", "")
            running = state.get("running_summary", "")
            context_prefix = ""
            if background:
                context_prefix += f"用户背景：{background}\n"
            if running:
                context_prefix += f"历史摘要：{running}\n"

            combined_context = "\n\n".join(sub_answers)
            if context_prefix:
                combined_context = context_prefix + "\n" + combined_context

            sys_tmpl = get_prompt_template("response")["system"]
            user_tmpl = get_prompt_template("response")["user"]
            synthesis_ai = llm.invoke(
                [
                    SystemMessage(content=sys_tmpl),
                    HumanMessage(
                        content=user_tmpl.format(
                            chat_history=_history_to_text(
                                state.get("dialogue_messages", [])
                            ),
                            all_document_str=combined_context,
                            input=state["curr_input"],
                        )
                    ),
                ]
            )
            final_answer = re.sub(
                r"<think>.*?</think>\s*", "", synthesis_ai.content, flags=re.DOTALL
            ).strip()

    # 更新对话历史
    state["final_answer"] = final_answer
    state["dialogue_messages"].append(HumanMessage(content=state["curr_input"]))
    state["dialogue_messages"].append(AIMessage(content=final_answer))

    # 更新检索查询展示
    sq = state.get("sub_query")
    if sq:
        if sq.need_split and sq.sub_query:
            state["rewritten_query"] = sq.sub_query[0]
        elif sq.rewrite_query:
            state["rewritten_query"] = sq.rewrite_query
        else:
            state["rewritten_query"] = state["curr_input"]

    # 记录本轮摘要（用于后续参考）
    short_summary = f"问：{state['curr_input']}\n答：{final_answer[:300]}"
    state["multi_summary"].append(short_summary)

    # 摘要压缩：当达到 8 条时，将最早 4 条压缩到 running_summary
    if len(state["multi_summary"]) >= 8:
        old_entries = state["multi_summary"][:4]
        summary_text = "\n".join(old_entries)
        compressed = llm.invoke(
            [
                SystemMessage(content=get_prompt_template("summary")["system"]),
                HumanMessage(
                    content=summary_text + "\n" + get_prompt_template("summary")["user"]
                ),
            ]
        )
        compressed_text = re.sub(
            r"<think>.*?</think>\s*", "", compressed.content, flags=re.DOTALL
        ).strip()
        prev = state.get("running_summary", "")
        state["running_summary"] = (
            (prev + "\n" + compressed_text).strip() if prev else compressed_text
        )
        # 移除已压缩的摘要
        state["multi_summary"] = state["multi_summary"][4:]

    # 保留最近 10 条摘要（压缩后通常不超过此数）
    if len(state["multi_summary"]) > 10:
        state["multi_summary"] = state["multi_summary"][-10:]

    # 注意：background_info 刻意保留（不清空），供下一轮 check_update_background 使用
    state["ask_obj"] = None

    return state


# ======================== MedicalAgent 主类 ========================


class MedicalAgent:
    """医学多轮对话 Agent

    集成追问、检索、LLM 合成等功能，支持多轮对话与历史压缩。
    """

    def __init__(self, config: AppConfig, power_model: BaseChatModel) -> None:
        """初始化 Agent

        Args:
            config: 应用配置
            power_model: 强力 LLM 模型（用于检索、分析）
        """
        self.config = config
        self.power_model = power_model
        self.normal_llm = create_llm_client(self.config.llm)
        self.search_graph = SearchGraph(self.config, power_model)
        self.app = None
        self.state: MedicalAgentState = {}
        self.build_graph()

    def build_graph(self):
        """构建 LangGraph 状态机"""
        g = StateGraph(MedicalAgentState)

        # 添加所有节点
        g.add_node("ask", partial(ask_judge, llm=self.normal_llm))
        g.add_node(
            "extract_ask_and_reply",
            partial(extract_background_info, llm=self.normal_llm),
        )
        g.add_node(
            "check_update_background",
            partial(check_update_background, llm=self.normal_llm),
        )
        g.add_node("condense_question", partial(condense_question, llm=self.normal_llm))
        g.add_node("split_query", partial(judge_split_query, llm=self.power_model))
        g.add_node("search_one", partial(search_one, search_graph=self.search_graph))
        g.add_node("answer", partial(gather_answer, llm=self.normal_llm))

        # 入口：START 条件路由
        g.add_conditional_edges(
            START,
            route_entry,
            {
                "ask": "ask",
                "check_update_background": "check_update_background",
            },
        )

        # 追问流程：route_ask_again 决定是否继续追问
        g.add_conditional_edges(
            "ask",
            route_ask_again,
            {
                "ask": END,  # 需要追问，结束本轮
                "pass": "extract_ask_and_reply",  # 信息充分，继续
            },
        )

        # 可选的问题改写环节
        if self.config.agent.enable_condense_question:
            g.add_edge("extract_ask_and_reply", "condense_question")
            g.add_edge("check_update_background", "condense_question")
            g.add_edge("condense_question", "split_query")
        else:
            g.add_edge("extract_ask_and_reply", "split_query")
            g.add_edge("check_update_background", "split_query")

        # 并行检索：split_query → 多个 search_one → answer
        g.add_conditional_edges("split_query", route_to_subgraphs, ["search_one"])
        g.add_edge("search_one", "answer")
        g.add_edge("answer", END)

        self.app = g.compile()
        self._reset_state()

    def _reset_state(self):
        """初始化/重置 Agent 状态"""
        self.state: MedicalAgentState = {
            "dialogue_messages": [],
            "asking_messages": [],
            "background_info": "",
            "ask_obj": None,
            "multi_summary": [],
            "running_summary": "",
            "rewritten_query": "",
            "curr_input": "",
            "sub_query": None,
            "sub_query_results": [],  # 每轮前重置，避免 add reducer 跨轮累积
            "max_ask_num": self.config.agent.max_ask_num,
            "curr_ask_num": 0,
            "final_answer": "",
            "performance": [],
            "show_debug": self.config.agent.console_debug,
        }

    def answer(self, user_input: str) -> MedicalAgentState:
        """处理用户输入，执行一轮完整流程

        Args:
            user_input: 用户输入文本

        Returns:
            更新后的完整状态（包含最终答案、追问消息等）
        """
        self.state["curr_input"] = user_input
        # 每轮前重置子查询结果，使 add reducer 从空列表开始
        self.state["sub_query_results"] = []
        self.state = self.app.invoke(self.state)
        return self.state
