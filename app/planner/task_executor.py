"""
任务执行器：按计划顺序执行子任务
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.rag import rag_system

from .models import TaskPlan, TaskMode, PlannedTask

logger = logging.getLogger(__name__)


class TaskExecutor:
    """执行 TaskPlan 并汇总最终答案。"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def _run_rag_task(self, task: PlannedTask) -> Dict[str, Any]:
        if not getattr(rag_system, "_initialized", False):
            logger.warning("Planner RAG 任务降级：RAG 未初始化")
            return {
                "task_id": task.id,
                "mode": task.mode.value,
                "objective": task.objective,
                "answer": "RAG 系统未初始化，无法检索。",
                "source_count": 0,
            }

        result = await rag_system.query(question=task.objective, with_sources=True)
        return {
            "task_id": task.id,
            "mode": task.mode.value,
            "objective": task.objective,
            "answer": result.get("answer", ""),
            "source_count": result.get("source_count", 0),
            "sources": result.get("sources", []),
        }

    async def _run_direct_task(
        self, original_question: str, task: PlannedTask, completed_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        context = json.dumps(
            [
                {
                    "task_id": x.get("task_id"),
                    "objective": x.get("objective"),
                    "answer": x.get("answer"),
                }
                for x in completed_results
            ],
            ensure_ascii=False,
        )
        messages = [
            SystemMessage(
                content=(
                    "你是任务执行助手。请根据已有结果完成当前子任务。"
                    "如果已有结果不足，请在不编造事实前提下给出保守回答。"
                )
            ),
            HumanMessage(
                content=(
                    f"原始问题：{original_question}\n"
                    f"当前任务：{task.objective}\n"
                    f"已完成任务结果：{context}\n\n"
                    "请直接输出当前任务答案。"
                )
            ),
        ]
        response = await self.llm.ainvoke(messages)
        answer = response.content if isinstance(response.content, str) else str(response.content)
        return {
            "task_id": task.id,
            "mode": task.mode.value,
            "objective": task.objective,
            "answer": answer,
            "source_count": 0,
        }

    async def _synthesize_final_answer(
        self, original_question: str, task_results: List[Dict[str, Any]]
    ) -> str:
        packed = json.dumps(task_results, ensure_ascii=False)
        messages = [
            SystemMessage(
                content=(
                    "你是结果整合助手。你会把多个子任务结果整合为最终答复。"
                    "要求：准确、简洁，不编造来源中不存在的事实。"
                )
            ),
            HumanMessage(
                content=(
                    f"原始问题：{original_question}\n"
                    f"子任务结果（JSON）：{packed}\n\n"
                    "请给出最终回答。"
                )
            ),
        ]
        response = await self.llm.ainvoke(messages)
        return response.content if isinstance(response.content, str) else str(response.content)

    async def aexecute(self, original_question: str, plan: TaskPlan) -> Dict[str, Any]:
        task_results: List[Dict[str, Any]] = []
        for task in sorted(plan.tasks, key=lambda x: x.order):
            if task.mode == TaskMode.RAG:
                result = await self._run_rag_task(task)
            else:
                result = await self._run_direct_task(
                    original_question=original_question,
                    task=task,
                    completed_results=task_results,
                )
            task_results.append(result)
            logger.info(
                "🧭 Executor 子任务完成: id=%s, mode=%s, source_count=%s",
                task.id,
                task.mode.value,
                result.get("source_count", 0),
            )

        final_answer = await self._synthesize_final_answer(
            original_question=original_question,
            task_results=task_results,
        )
        return {
            "final_answer": final_answer,
            "task_results": task_results,
        }
