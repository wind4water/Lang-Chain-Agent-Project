"""
向量存储管理
"""
from typing import Optional, List
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import os
import shutil
import logging

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """向量存储管理器（Chroma）"""

    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: str = "data/vectordb/chroma",
        collection_name: str = "default_collection"
    ):
        """
        初始化向量存储管理器

        Args:
            embeddings: 嵌入模型
            persist_directory: 持久化目录
            collection_name: 集合名称
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._vectorstore = None

        # 确保目录存在
        os.makedirs(persist_directory, exist_ok=True)

    @property
    def vectorstore(self) -> Chroma:
        """获取向量存储实例（懒加载）"""
        if self._vectorstore is None:
            logger.info(f"加载向量存储: {self.collection_name}")
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        return self._vectorstore

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        添加文档到向量存储

        Args:
            documents: 文档列表

        Returns:
            文档 ID 列表
        """
        if not documents:
            logger.warning("没有文档需要添加")
            return []

        logger.info(f"添加 {len(documents)} 个文档到向量存储")
        ids = self.vectorstore.add_documents(documents)
        logger.info(f"✅ 成功添加 {len(ids)} 个文档")
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回文档数量
            filter: 过滤条件

        Returns:
            相关文档列表
        """
        logger.info(f"相似度搜索: query='{query}', k={k}")
        results = self.vectorstore.similarity_search(
            query,
            k=k,
            filter=filter
        )
        logger.info(f"检索到 {len(results)} 个相关文档")
        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[tuple[Document, float]]:
        """
        相似度搜索（带分数）

        Args:
            query: 查询文本
            k: 返回文档数量
            filter: 过滤条件

        Returns:
            (文档, 相似度分数) 元组列表
        """
        logger.info(f"相似度搜索（带分数）: query='{query}', k={k}")
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filter
        )
        logger.info(f"检索到 {len(results)} 个相关文档")
        return results

    def clear(self):
        """清空向量存储"""
        logger.warning(f"清空向量存储: {self.collection_name}")
        try:
            # 删除集合
            if self._vectorstore is not None:
                self._vectorstore.delete_collection()
                self._vectorstore = None

            # 删除持久化目录
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                os.makedirs(self.persist_directory, exist_ok=True)

            logger.info("✅ 向量存储已清空")
        except Exception as e:
            logger.error(f"清空向量存储失败: {e}")
            raise

    def rebuild(self, documents: List[Document]) -> List[str]:
        """
        重建向量存储

        Args:
            documents: 文档列表

        Returns:
            文档 ID 列表
        """
        logger.info(f"重建向量存储: {len(documents)} 个文档")

        # 清空现有数据
        self.clear()

        # 添加新文档
        ids = self.add_documents(documents)

        logger.info(f"✅ 向量存储重建完成: {len(ids)} 个文档")
        return ids

    def get_retriever(
        self,
        search_type: str = "similarity",
        k: int = 4,
        score_threshold: float = 0.5
    ):
        """
        获取检索器

        Args:
            search_type: 搜索类型 (similarity, mmr, similarity_score_threshold)
            k: 返回文档数量
            score_threshold: 相似度阈值（仅用于 similarity_score_threshold）

        Returns:
            检索器实例
        """
        search_kwargs = {"k": k}

        if search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = score_threshold

        retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

        logger.info(f"创建检索器: search_type={search_type}, k={k}")
        return retriever

    def get_stats(self) -> dict:
        """
        获取向量存储统计信息

        Returns:
            统计信息字典
        """
        try:
            collection = self.vectorstore._collection
            count = collection.count()

            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": os.path.abspath(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "error": str(e)
            }
