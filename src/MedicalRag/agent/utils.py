"""
Agent 工具函数

包含消息处理、Token统计等辅助功能
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import re

if TYPE_CHECKING:
    from .MedicalAgent import MedicalAgentState

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


def get_last_human(state: MedicalAgentState, stage: str = "asking_messages") -> str:
    """获取最后一条用户消息

    Args:
        state: Agent 状态
        stage: 消息来源（"asking_messages" 或 "dialogue"）

    Returns:
        最后一条用户消息内容，没有则返回空字符串
    """
    mess = "dialogue_messages" if stage == "dialogue" else "asking_messages"
    for m in reversed(state.get(mess, [])):
        if isinstance(m, HumanMessage) and getattr(m, "content", ""):
            return m.content
    return ""


def strip_think_get_tokens(msg: AIMessage) -> dict:
    """处理 LLM 响应：移除思考标签、提取 Token 统计

    一些支持思考的模型（如 DeepSeek）会返回 <think>...</think> 标签，
    需要移除后才是实际回复内容。同时提取 Token 使用情况用于性能分析。

    Args:
        msg: LLM 返回的消息对象

    Returns:
        {
            "msg": 清理后的消息文本,
            "msg_len": 原文本长度,
            "msg_token_len": 输出 Token 数,
            "generate_time": 生成耗时（秒）
        }
    """
    text = msg.content
    msg_len = len(msg.content)

    # 尝试从多个地方获取 Token 统计
    msg_token_len = 0
    try:
        msg_token_len = msg.usage_metadata["output_tokens"]
    except Exception:
        try:
            msg_token_len = msg.response_metadata["token_usage"]["output_tokens"]
        except Exception:
            msg_token_len = 0

    # 获取生成耗时（单位：秒）
    dur = msg.response_metadata.get("total_duration", 0) / 1e9

    # 移除 <think>...</think> 标签
    clean_text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    return {
        "msg": clean_text,
        "msg_len": msg_len,
        "msg_token_len": msg_token_len,
        "generate_time": dur,
    }
