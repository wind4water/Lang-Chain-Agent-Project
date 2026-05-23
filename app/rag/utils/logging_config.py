"""
RAG 模块日志配置

提供统一的日志格式和配置管理
"""
import logging
import os
from typing import Optional


def setup_rag_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    enable_rich: bool = True
) -> None:
    """
    配置 RAG 模块的日志系统

    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR)，默认从环境变量获取
        format_string: 自定义格式字符串
        enable_rich: 是否启用 rich 格式化输出（如果有 rich 库）
    """
    level = level or os.getenv("RAG_LOG_LEVEL", "INFO")
    
    # 默认格式
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
        )
    
    # 配置根 logger
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format_string))
    
    # 设置 RAG 模块的日志级别
    rag_logger = logging.getLogger("app.rag")
    rag_logger.setLevel(getattr(logging, level.upper()))
    rag_logger.handlers = []
    rag_logger.addHandler(handler)
    
    # 防止重复日志
    rag_logger.propagate = False


class RAGLogLevel:
    """日志级别常量"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class RAGLogEvent:
    """标准日志事件类型"""
    # 系统生命周期
    SYSTEM_INIT_START = "system_init_start"
    SYSTEM_INIT_COMPLETE = "system_init_complete"
    SYSTEM_INIT_FAILED = "system_init_failed"
    
    # 嵌入模型
    EMBEDDING_LOAD_START = "embedding_load_start"
    EMBEDDING_LOAD_COMPLETE = "embedding_load_complete"
    EMBEDDING_LOAD_FAILED = "embedding_load_failed"
    
    # 向量存储
    VECTORSTORE_INIT = "vectorstore_init"
    VECTORSTORE_CLEAR = "vectorstore_clear"
    VECTORSTORE_ADD = "vectorstore_add"
    VECTORSTORE_DELETE = "vectorstore_delete"
    
    # 文档处理
    DOCUMENT_LOAD_START = "document_load_start"
    DOCUMENT_LOAD_COMPLETE = "document_load_complete"
    DOCUMENT_LOAD_FAILED = "document_load_failed"
    DOCUMENT_SPLIT = "document_split"
    
    # 检索
    RETRIEVE_START = "retrieve_start"
    RETRIEVE_COMPLETE = "retrieve_complete"
    RETRIEVE_FAILED = "retrieve_failed"
    RETRIEVE_RRF_FUSION = "retrieve_rrf_fusion"
    
    # 查询
    QUERY_START = "query_start"
    QUERY_COMPLETE = "query_complete"
    QUERY_FAILED = "query_failed"
    
    # 重建/同步
    REBUILD_START = "rebuild_start"
    REBUILD_COMPLETE = "rebuild_complete"
    SYNC_START = "sync_start"
    SYNC_COMPLETE = "sync_complete"
