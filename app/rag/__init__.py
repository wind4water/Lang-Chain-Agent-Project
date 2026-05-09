"""
RAG 模块 - 检索增强生成

提供基于向量数据库的知识库问答功能
"""

from app.rag.system import rag_system, RAGSystem
from app.rag.config import rag_config, RAGConfig

__all__ = [
    "rag_system",
    "RAGSystem",
    "rag_config",
    "RAGConfig",
]
