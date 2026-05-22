"""
Task planner 数据模型
"""
from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class TaskMode(str, Enum):
    """子任务执行模式。"""

    RAG = "rag"
    DIRECT = "direct"


class PlannedTask(BaseModel):
    """单个计划任务。"""

    id: str = Field(default="", description="任务唯一ID（可省略，系统会自动补齐）")
    objective: str = Field(description="该子任务要完成的具体目标")
    mode: TaskMode = Field(description="执行模式：rag/direct")
    order: int = Field(default=1, description="执行顺序，从1开始")


class TaskPlan(BaseModel):
    """结构化任务规划结果。"""

    tasks: List[PlannedTask] = Field(default_factory=list, description="任务列表")
