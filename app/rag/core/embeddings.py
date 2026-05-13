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
    """嵌入模型管理器"""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        初始化嵌入模型管理器

        Args:
            model_name: 模型名称
            api_key: API 密钥
            base_url: API 基础 URL
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY 未配置")

        self._embeddings = None

    @property
    def embeddings(self) -> Embeddings:
        """获取嵌入模型实例（懒加载）"""
        if self._embeddings is None:
            logger.info(f"初始化嵌入模型: {self.model_name}")
            emb_kwargs = {
                "model": self.model_name,
                "api_key": self.api_key,
                "base_url": self.base_url,
            }
            if not _SSL_VERIFY:
                emb_kwargs["http_client"] = _http_client
            self._embeddings = OpenAIEmbeddings(**emb_kwargs)
        return self._embeddings

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
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            # 智谱AI嵌入模型
            "embedding-2": 1024,
            "embedding-3": 1024,  # 可配置256-2048，默认1024
        }
        return dimensions.get(self.model_name, 1536)
