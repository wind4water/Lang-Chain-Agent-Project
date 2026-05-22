"""
基于 LLM 的任务规划器
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_openai import ChatOpenAI

from .models import TaskPlan, PlannedTask

logger = logging.getLogger(__name__)


class TaskPlanner:
    """将用户问题规划成结构化子任务。"""

    def __init__(self, llm: ChatOpenAI, max_tasks: int = 5):
        self.llm = llm
        self.max_tasks = max_tasks
        self._structured_llm = self.llm.with_structured_output(TaskPlan)
        self._force_rag_keywords = [
            "公司", "员工", "同事", "某个人", "谁", "人物", "个人",
            "路由", "支付", "金融", "银行", "账户", "清结算", "交易",
        ]

    def _requires_rag(self, user_query: str) -> bool:
        text = (user_query or "").lower()
        return any(k in text for k in self._force_rag_keywords)

    @staticmethod
    def _build_prompt(user_query: str, max_tasks: int, force_rag: bool) -> str:
        extra_rule = (
            "6) 如果问题涉及公司、某个人、员工、路由、支付、金融等领域事实，"
            "任务列表中至少包含 1 个 mode=rag 的任务。\n"
            if force_rag
            else ""
        )
        return (
            "你是任务规划智能体。请将用户问题拆解为可执行子任务。\n"
            "要求：\n"
            "1) 拆解成若干子任务，并给出执行顺序 order（从1开始）；\n"
            "2) 每个任务必须标注 mode: rag 或 direct；\n"
            "3) mode=rag 用于需要知识库检索事实的问题；mode=direct 用于无需检索的总结/润色/整合；\n"
            f"4) 任务数量不超过 {max_tasks}；\n"
            "5) objective 保持具体、可执行，避免过泛。\n"
            f"{extra_rule}\n"
            f"用户问题：{user_query}"
        )

    async def aplan(self, user_query: str) -> Optional[TaskPlan]:
        """异步生成任务计划。"""
        force_rag = self._requires_rag(user_query)
        prompt = self._build_prompt(
            user_query=user_query,
            max_tasks=self.max_tasks,
            force_rag=force_rag,
        )
        try:
            plan = await self._structured_llm.ainvoke(prompt)
            if not isinstance(plan, TaskPlan):
                return None

            tasks = sorted(plan.tasks, key=lambda x: x.order)
            tasks = tasks[: self.max_tasks]
            normalized_tasks = []
            for idx, task in enumerate(tasks, 1):
                normalized_tasks.append(
                    PlannedTask(
                        id=task.id or f"task_{idx}",
                        objective=(task.objective or "").strip(),
                        mode=task.mode,
                        order=idx,
                    )
                )

            # 兜底：涉及事实型关键词但模型仍未给 rag 任务时，强制将首个任务改为 rag。
            if force_rag and normalized_tasks and not any(t.mode.value == "rag" for t in normalized_tasks):
                first = normalized_tasks[0]
                normalized_tasks[0] = PlannedTask(
                    id=first.id,
                    objective=first.objective,
                    mode="rag",
                    order=first.order,
                )
                logger.info("🧭 Planner 兜底修正: 已将首个任务强制设为 mode=rag")

            normalized_plan = TaskPlan(tasks=normalized_tasks)
            logger.info("🧭 Planner 生成计划: tasks=%s", len(normalized_plan.tasks))
            return normalized_plan
        except Exception as e:
            logger.warning("Planner 生成失败，回退默认流程: %s", e)
            return None
