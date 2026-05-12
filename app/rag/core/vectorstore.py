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
import time
import stat

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

        # 确保目录存在且可写
        self._ensure_persist_directory_writable()

    def _ensure_persist_directory_writable(self):
        """确保持久化目录存在且当前进程可写。"""
        os.makedirs(self.persist_directory, exist_ok=True)

        # 尝试修复目录权限（在当前用户有权限时有效）
        try:
            current_mode = stat.S_IMODE(os.stat(self.persist_directory).st_mode)
            expected_mode = current_mode | stat.S_IWUSR | stat.S_IXUSR
            if expected_mode != current_mode:
                os.chmod(self.persist_directory, expected_mode)
        except OSError as e:
            logger.warning(f"修复目录权限失败: {self.persist_directory}, {e}")

        if not os.access(self.persist_directory, os.W_OK | os.X_OK):
            raise PermissionError(
                f"向量库目录无写权限: {os.path.abspath(self.persist_directory)}"
            )

    def _new_vectorstore_instance(self) -> Chroma:
        """创建新的 Chroma 实例。"""
        self._ensure_persist_directory_writable()
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    @property
    def vectorstore(self) -> Chroma:
        """获取向量存储实例（懒加载）"""
        if self._vectorstore is None:
            logger.info(f"加载向量存储: {self.collection_name}")
            self._vectorstore = self._new_vectorstore_instance()
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

    def delete_by_source(self, source_path: str) -> int:
        """
        根据文档来源删除文档

        Args:
            source_path: 文档来源路径（在metadata中的source字段）

        Returns:
            删除的文档数量
        """
        try:
            collection = self.vectorstore._collection
            # 查询所有匹配的文档ID
            results = collection.get(where={"source": source_path})

            if not results or not results.get('ids'):
                logger.debug(f"未找到来源为 {source_path} 的文档")
                return 0

            ids_to_delete = results['ids']
            collection.delete(ids=ids_to_delete)

            logger.info(f"✅ 删除了 {len(ids_to_delete)} 个文档块（来源: {source_path}）")
            return len(ids_to_delete)

        except Exception as e:
            logger.error(f"删除文档失败 {source_path}: {e}")
            return 0

    def update_documents(self, documents: List[Document], source_path: str) -> List[str]:
        """
        更新文档（先删除旧版本，再添加新版本）

        Args:
            documents: 新的文档列表
            source_path: 文档来源路径

        Returns:
            新文档的 ID 列表
        """
        logger.info(f"更新文档: {source_path}")

        # 删除旧版本
        deleted_count = self.delete_by_source(source_path)
        logger.debug(f"删除旧文档: {deleted_count} 个块")

        # 添加新版本
        ids = self.add_documents(documents)
        logger.info(f"✅ 更新完成: {source_path} ({len(ids)} 个块)")

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
            deleted_by_api = False

            # 优先通过 Chroma API 删除集合，避免直接删目录导致 sqlite 文件句柄状态异常
            target_store = self._vectorstore
            if target_store is None:
                try:
                    target_store = self._new_vectorstore_instance()
                except Exception as e:
                    logger.warning(f"创建临时向量存储实例失败，将尝试目录级清理: {e}")

            if target_store is not None:
                try:
                    target_store.delete_collection()
                    deleted_by_api = True
                    logger.info(f"已删除集合: {self.collection_name}")
                except Exception as e:
                    logger.warning(f"删除集合失败（可能不存在）: {e}")
                finally:
                    self._vectorstore = None

            # 等待一小段时间，让 Chroma 释放文件句柄
            time.sleep(0.2)

            # 仅在 API 删除失败时，才回退到目录级清理
            if not deleted_by_api and os.path.exists(self.persist_directory):
                logger.info(f"集合删除失败，回退到目录级清理: {self.persist_directory}")
                try:
                    shutil.rmtree(self.persist_directory)
                    logger.info("目录已删除")
                except PermissionError as e:
                    logger.error(f"删除目录失败（权限错误）: {e}")
                    logger.info("尝试使用替代方案：重命名旧目录")
                    backup_dir = f"{self.persist_directory}_old_{int(time.time())}"
                    os.rename(self.persist_directory, backup_dir)
                    logger.warning(f"旧目录已重命名为: {backup_dir}（请手动删除）")

            self._ensure_persist_directory_writable()
            logger.info("目录可写性检查通过")

            logger.info("✅ 向量存储已彻底清空")
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
        score_threshold: float = 0.5,
        metadata_filter: Optional[dict] = None
    ):
        """
        获取检索器

        Args:
            search_type: 搜索类型 (similarity, mmr, similarity_score_threshold)
            k: 返回文档数量
            score_threshold: 相似度阈值（仅用于 similarity_score_threshold）
            metadata_filter: 元数据过滤条件（传递给 Chroma filter）

        Returns:
            检索器实例
        """
        search_kwargs = {"k": k}

        if search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = score_threshold
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter

        retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

        logger.info(
            f"创建检索器: search_type={search_type}, k={k}, "
            f"metadata_filter={'yes' if metadata_filter else 'no'}"
        )
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
