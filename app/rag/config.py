"""
RAG 模块配置
"""
import os
from typing import Optional
from pydantic import BaseModel, Field


class RAGConfig(BaseModel):
    """RAG 配置"""

    # 日志配置
    rag_log_level: str = Field(
        default_factory=lambda: os.getenv("RAG_LOG_LEVEL", "INFO"),
        description="RAG 日志级别: DEBUG, INFO, WARNING, ERROR"
    )
    rag_verbose: bool = Field(
        default_factory=lambda: os.getenv("RAG_VERBOSE", "false").lower() == "true",
        description="是否启用详细日志"
    )
    
    # 嵌入模型配置
    embedding_type: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_TYPE", "local"),
        description="嵌入类型: local（本地模型）, remote（远程API）"
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "bge-base-zh-v1.5"),
        description="嵌入模型名称"
    )
    embedding_device: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"),
        description="本地模型设备: cpu, cuda"
    )
    embedding_dimension: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "768")),
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
        default_factory=lambda: os.getenv("RAG_DOCUMENTS_PATH", "./data/documents,./data/projects"),
        description="文档目录路径（支持多个路径，用逗号分隔）"
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

    # 多模型配置
    multi_model_config: str = Field(
        default_factory=lambda: os.getenv("MULTI_MODEL_CONFIG", ""),
        description="多模型配置（JSON格式）"
    )
    doc_model_weight: float = Field(
        default_factory=lambda: float(os.getenv("DOC_MODEL_WEIGHT", "0.5")),
        description="文档模型检索权重"
    )
    code_model_weight: float = Field(
        default_factory=lambda: float(os.getenv("CODE_MODEL_WEIGHT", "0.5")),
        description="代码模型检索权重"
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

    # HNSW 配置
    hnsw_space: str = Field(
        default_factory=lambda: os.getenv("HNSW_SPACE", "l2"),
        description="HNSW 距离度量: cosine, l2, ip"
    )
    hnsw_m: int = Field(
        default_factory=lambda: int(os.getenv("HNSW_M", "16")),
        description="HNSW 最大连接数"
    )
    hnsw_ef_search: int = Field(
        default_factory=lambda: int(os.getenv("HNSW_EF_SEARCH", "100")),
        description="HNSW 搜索候选列表大小"
    )

    def get_collection_metadata(self) -> dict:
        """
        获取 HNSW collection 配置元数据
        
        Returns:
            ChromaDB collection 创建时的 metadata
        """
        return {
            "hnsw:space": self.hnsw_space,
            "hnsw:construction_ef": 200,
            "hnsw:search_ef": self.hnsw_ef_search,
            "hnsw:M": self.hnsw_m,
        }

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

    def get_vector_store_path(self) -> str:
        """
        根据 embedding 配置生成专属的向量存储路径

        格式：{base_path}/{embedding_type}_{model_name}/
        例如：
        - ./data/vectordb/chroma/local_bge-base-zh-v1.5/
        - ./data/vectordb/chroma/remote_text-embedding-3-small/

        Returns:
            向量存储路径
        """
        # 清理模型名称，替换特殊字符
        clean_model_name = self.embedding_model.replace("/", "_").replace(":", "_")

        # 生成专属目录名
        dir_name = f"{self.embedding_type}_{clean_model_name}"

        # 拼接完整路径
        base_path = self.chroma_path.rstrip("/")
        return f"{base_path}/{dir_name}"

    def get_registry_path(self) -> str:
        """
        根据 embedding 配置生成专属的文档注册表路径

        格式：{base_path}/registry/{embedding_type}_{model_name}.json
        例如：
        - ./data/vectordb/registry/local_bge-base-zh-v1.5.json
        - ./data/vectordb/registry/remote_text-embedding-3-small.json

        Returns:
            文档注册表路径
        """
        # 清理模型名称
        clean_model_name = self.embedding_model.replace("/", "_").replace(":", "_")

        # 生成专属文件名
        file_name = f"{self.embedding_type}_{clean_model_name}.json"

        # 拼接完整路径（放在 vectordb 目录下的 registry 子目录）
        base_path = os.path.dirname(self.chroma_path.rstrip("/"))
        return f"{base_path}/registry/{file_name}"


# 全局配置实例
rag_config = RAGConfig()
