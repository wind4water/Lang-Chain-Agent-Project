"""
RAG 模块配置
"""
import os
from typing import Optional
from pydantic import BaseModel, Field


class RAGConfig(BaseModel):
    """RAG 配置"""

    # 嵌入模型配置
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "embedding-3"),
        description="嵌入模型名称"
    )
    embedding_dimension: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "1024")),
        description="嵌入向量维度"
    )

    # 向量数据库配置
    vector_store_type: str = Field(
        default_factory=lambda: os.getenv("VECTOR_STORE", "chroma"),
        description="向量数据库类型"
    )
    chroma_path: str = Field(
        default_factory=lambda: os.getenv("CHROMA_PATH", "./data/vectordb/chroma"),
        description="Chroma 数据库路径"
    )
    collection_name: str = Field(
        default_factory=lambda: os.getenv("RAG_COLLECTION_NAME", "default_collection"),
        description="集合名称"
    )

    # 文档路径配置
    documents_path: str = Field(
        default_factory=lambda: os.getenv("RAG_DOCUMENTS_PATH", "./data/documents"),
        description="文档目录路径"
    )

    # 检索配置
    top_k: int = Field(
        default_factory=lambda: int(os.getenv("RAG_TOP_K", "4")),
        description="检索返回的文档数量"
    )
    search_type: str = Field(
        default_factory=lambda: os.getenv("RAG_SEARCH_TYPE", "similarity"),
        description="搜索类型: similarity, mmr, similarity_score_threshold"
    )
    retrieval_mode: str = Field(
        default_factory=lambda: os.getenv("RAG_RETRIEVAL_MODE", "vector"),
        description="检索模式: vector, hybrid"
    )
    keyword_backend: str = Field(
        default_factory=lambda: os.getenv("RAG_KEYWORD_BACKEND", "local"),
        description="关键词召回后端: local, es"
    )
    score_threshold: float = Field(
        default_factory=lambda: float(os.getenv("RAG_SCORE_THRESHOLD", "0.5")),
        description="相似度阈值"
    )
    hybrid_keyword_k: int = Field(
        default_factory=lambda: int(os.getenv("RAG_HYBRID_KEYWORD_K", "8")),
        description="Hybrid 模式下关键词召回候选数量"
    )

    # Elasticsearch 关键词召回配置
    es_enabled: bool = Field(
        default_factory=lambda: os.getenv("ES_ENABLED", "false").lower() == "true",
        description="是否启用 Elasticsearch 关键词召回"
    )
    es_hosts: str = Field(
        default_factory=lambda: os.getenv("ES_HOSTS", "http://127.0.0.1:9200"),
        description="ES 地址，多个地址用逗号分隔"
    )
    es_index_name: str = Field(
        default_factory=lambda: os.getenv("ES_INDEX_NAME", "rag_keyword_chunks"),
        description="ES 索引名称"
    )
    es_username: Optional[str] = Field(
        default_factory=lambda: os.getenv("ES_USERNAME"),
        description="ES 用户名（可选）"
    )
    es_password: Optional[str] = Field(
        default_factory=lambda: os.getenv("ES_PASSWORD"),
        description="ES 密码（可选）"
    )
    es_verify_certs: bool = Field(
        default_factory=lambda: os.getenv("ES_VERIFY_CERTS", "true").lower() == "true",
        description="ES HTTPS 证书校验"
    )
    es_ca_certs: Optional[str] = Field(
        default_factory=lambda: os.getenv("ES_CA_CERTS"),
        description="ES CA 证书路径（可选）"
    )
    es_request_timeout: int = Field(
        default_factory=lambda: int(os.getenv("ES_REQUEST_TIMEOUT", "30")),
        description="ES 请求超时时间（秒）"
    )

    # 文本分割配置
    chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")),
        description="文本块大小"
    )
    chunk_overlap: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")),
        description="文本块重叠大小"
    )

    # 是否在启动时重建索引
    rebuild_on_startup: bool = Field(
        default_factory=lambda: os.getenv("RAG_REBUILD_ON_STARTUP", "true").lower() == "true",
        description="启动时是否重建知识库"
    )

    # 是否启用 RAG
    enabled: bool = Field(
        default_factory=lambda: os.getenv("RAG_ENABLED", "false").lower() == "true",
        description="是否启用 RAG 功能"
    )

    class Config:
        arbitrary_types_allowed = True


# 全局配置实例
rag_config = RAGConfig()
