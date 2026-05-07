"""
向后兼容层 - agent.py
实际实现已移至 app/agents/memory.py
"""
from app.agents.memory import MemoryAgent as ConversationAgent

# 保持旧的导入方式可用
__all__ = ['ConversationAgent']
