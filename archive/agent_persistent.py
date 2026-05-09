"""
向后兼容层 - agent_persistent.py
实际实现已移至 app/agents/sqlite.py
"""
from app.agents.sqlite import SqliteAgent as ConversationAgentWithPersistence

# 保持旧的导入方式可用
__all__ = ['ConversationAgentWithPersistence']
