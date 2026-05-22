"""
嵌入模型管理
"""
from typing import Optional
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
import os
import logging
import httpx
import urllib3

logger = logging.getLogger(__name__)

# SSL 验证配置（默认启用）
_SSL_VERIFY = os.getenv("SSL_VERIFY", "true").lower() != "false"
_http_client = None
if not _SSL_VERIFY:
    logger.warning("⚠️ SSL 证书验证已禁用（SSL_VERIFY=false）")
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _http_client = httpx.Client(verify=False)


class EmbeddingManager:
    """嵌入模型管理器 - 支持本地和远程 Embedding"""

    def __init__(
        self,
        model_name: str = "bge-base-zh-v1.5",
        embedding_type: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        chunk_size: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        初始化嵌入模型管理器

        Args:
            model_name: 模型名称
            embedding_type: Embedding 类型 (local/remote)
            api_key: API 密钥（仅 remote 模式需要）
            base_url: API 基础 URL（仅 remote 模式需要）
            chunk_size: 批次大小（仅 remote 模式需要）
            device: 设备类型 (cpu/cuda)，仅本地模式需要
        """
        self.embedding_type = embedding_type or os.getenv("EMBEDDING_TYPE", "local")
        self.model_name = model_name
        self.device = device or os.getenv("EMBEDDING_DEVICE", "cpu")

        # 远程 API 配置
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.chunk_size = chunk_size or int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

        # 验证配置
        if self.embedding_type == "remote" and not self.api_key:
            raise ValueError("❌ 远程模式需要配置 OPENAI_API_KEY")

        self._embeddings = None

    @property
    def embeddings(self) -> Embeddings:
        """获取嵌入模型实例（懒加载）"""
        if self._embeddings is None:
            if self.embedding_type == "local":
                self._embeddings = self._init_local_embeddings()
            else:
                self._embeddings = self._init_remote_embeddings()
        return self._embeddings

    def _init_local_embeddings(self) -> Embeddings:
        """初始化本地 Embedding 模型"""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError(
                "本地 Embedding 需要安装 langchain-huggingface 和 sentence-transformers\n"
                "请运行: pip install langchain-huggingface sentence-transformers"
            )

        logger.info(f"🔧 初始化本地嵌入模型: {self.model_name} (device={self.device})")

        # 支持的本地模型映射
        model_map = {
            "bge-base-zh-v1.5": "BAAI/bge-base-zh-v1.5",
            "bge-large-zh-v1.5": "BAAI/bge-large-zh-v1.5",
            "text2vec-base-chinese": "shibing624/text2vec-base-chinese",
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        }

        # 获取完整模型名称
        full_model_name = model_map.get(self.model_name, self.model_name)

        embeddings = HuggingFaceEmbeddings(
            model_name=full_model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )

        logger.info(f"✅ 本地嵌入模型加载成功")
        logger.info(f"   - 模型: {full_model_name}")
        logger.info(f"   - 设备: {self.device}")
        logger.info(f"   - 缓存位置: ~/.cache/huggingface/hub/")

        return embeddings

    def _init_remote_embeddings(self) -> Embeddings:
        """初始化远程 API Embedding 模型"""
        logger.info(f"🌐 初始化远程嵌入模型: {self.model_name}")
        emb_kwargs = {
            "model": self.model_name,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "chunk_size": self.chunk_size,
        }
        if not _SSL_VERIFY:
            emb_kwargs["http_client"] = _http_client

        embeddings = OpenAIEmbeddings(**emb_kwargs)
        logger.info(f"✅ 远程嵌入模型配置成功")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        嵌入单个查询文本

        Args:
            text: 查询文本

        Returns:
            嵌入向量
        """
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量嵌入文档

        Args:
            texts: 文档列表

        Returns:
            嵌入向量列表
        """
        return self.embeddings.embed_documents(texts)

    def get_dimension(self) -> int:
        """
        获取嵌入向量维度

        Returns:
            向量维度
        """
        # 不同模型的维度
        dimensions = {
            # 本地模型
            "bge-base-zh-v1.5": 768,
            "bge-large-zh-v1.5": 1024,
            "text2vec-base-chinese": 768,
            "all-MiniLM-L6-v2": 384,
            "paraphrase-multilingual-MiniLM-L12-v2": 384,
            # OpenAI 模型
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            # 智谱AI嵌入模型
            "embedding-2": 1024,
            "embedding-3": 1024,  # 可配置256-2048，默认1024
        }
        return dimensions.get(self.model_name, 768)

    def get_info(self) -> dict:
        """
        获取 Embedding 配置信息

        Returns:
            配置信息字典
        """
        return {
            "type": self.embedding_type,
            "model": self.model_name,
            "dimension": self.get_dimension(),
            "device": self.device if self.embedding_type == "local" else "N/A",
            "api_endpoint": self.base_url if self.embedding_type == "remote" else "N/A",
        }
