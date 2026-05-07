"""
向后兼容层 - agent_postgres.py
实际实现已移至 app/agents/postgres.py
"""
from app.agents.postgres import PostgresAgent as ConversationAgentPostgres

# 保持旧的导入方式可用
__all__ = ['ConversationAgentPostgres']
