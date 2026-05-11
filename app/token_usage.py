"""
请求级 Token 统计工具
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Dict, Any, Optional


_request_token_usage: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "request_token_usage",
    default=None
)


@asynccontextmanager
async def request_token_scope():
    """
    为单次 HTTP 请求创建 token 统计上下文。
    """
    token = _request_token_usage.set(
        {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model_calls": 0,
        }
    )
    try:
        yield
    finally:
        _request_token_usage.reset(token)


def record_llm_token_usage(prompt_tokens: int = 0, completion_tokens: int = 0, model_name: str = ""):
    """
    记录一次 LLM 调用的 token 使用量。
    如果当前不在请求作用域内，则静默忽略。
    """
    usage = _request_token_usage.get()
    if usage is None:
        return

    prompt = max(int(prompt_tokens or 0), 0)
    completion = max(int(completion_tokens or 0), 0)

    usage["prompt_tokens"] += prompt
    usage["completion_tokens"] += completion
    usage["total_tokens"] += (prompt + completion)
    usage["model_calls"] += 1


def get_request_token_usage() -> Dict[str, int]:
    """
    获取当前请求的累计 token 统计。
    """
    usage = _request_token_usage.get()
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model_calls": 0,
        }

    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "model_calls": int(usage.get("model_calls", 0)),
    }

