"""
双模型混合检索器
支持 BGE（通用）+ CodeBERT（代码）双路召回
"""
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import logging

from app.rag.config import rag_config
from app.rag.core.code_embedding_manager import CodeEmbeddingManager

logger = logging.getLogger(__name__)


class DualModelRetriever:
    """双模型混合检索器"""

    def __init__(
        self,
        doc_retriever,      # BGE 检索器（文档）
        code_retriever=None, # CodeBERT 检索器（代码）
        doc_weight: float = 0.5,  # 文档结果权重
        code_weight: float = 0.5,  # 代码结果权重
        k: int = 5
    ):
        """
        初始化双模型检索器

        Args:
            doc_retriever: 文档检索器（BGE）
            code_retriever: 代码检索器（CodeBERT），可选
            doc_weight: 文档结果权重
            code_weight: 代码结果权重
            k: 最终结果数
        """
        self.doc_retriever = doc_retriever
        self.code_retriever = code_retriever
        self.doc_weight = doc_weight
        self.code_weight = code_weight
        self.k = k

        self.code_enabled = (
            rag_config.code_embedding_enabled
            and code_retriever is not None
        )

        if self.code_enabled:
            logger.info(
                f"双模型检索已启用: doc_weight={doc_weight}, "
                f"code_weight={code_weight}"
            )
        else:
            logger.info("单模型检索模式（文档）")

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        执行双路检索并合并结果

        Args:
            query: 查询文本

        Returns:
            合并后的文档列表
        """
        if not self.code_enabled:
            # 单模型模式
            return self.doc_retriever.get_relevant_documents(query)

        # 双路检索
        logger.info(f"双路检索: {query[:50]}...")

        # 1. BGE 检索文档
        doc_results = self.doc_retriever.get_relevant_documents(query)
        logger.info(f"文档检索: {len(doc_results)} 条")

        # 2. CodeBERT 检索代码
        code_results = self.code_retriever.get_relevant_documents(query)
        logger.info(f"代码检索: {len(code_results)} 条")

        # 3. 合并结果
        merged = self._merge_results(doc_results, code_results)
        logger.info(f"合并结果: {len(merged)} 条")

        return merged[:self.k]

    def _merge_results(
        self,
        doc_results: List[Document],
        code_results: List[Document]
    ) -> List[Document]:
        """
        合并两路结果，去重并加权排序
        """
        from collections import OrderedDict

        # 给结果加上权重标记
        weighted_docs = []

        for doc in doc_results:
            doc.metadata['_retriever'] = 'doc'
            doc.metadata['_score'] = doc.metadata.get('score', 0.5) * self.doc_weight
            weighted_docs.append((doc.metadata.get('score', 0), doc))

        for doc in code_results:
            doc.metadata['_retriever'] = 'code'
            doc.metadata['_score'] = doc.metadata.get('score', 0.5) * self.code_weight
            weighted_docs.append((doc.metadata.get('score', 0), doc))

        # 按分数排序
        weighted_docs.sort(key=lambda x: x[0], reverse=True)

        # 去重（基于 source + 内容前100字符）
        seen = set()
        unique_docs = []
        for _, doc in weighted_docs:
            key = f"{doc.metadata.get('source', '')}_{doc.page_content[:100]}"
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        return unique_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """异步版本"""
        return self.get_relevant_documents(query)


class DualModelHybridRetriever(DualModelRetriever):
    """双模型 + Hybrid 检索器（支持关键词）"""

    def __init__(
        self,
        doc_retriever,
        code_retriever=None,
        keyword_retriever=None,
        doc_weight: float = 0.4,
        code_weight: float = 0.4,
        keyword_weight: float = 0.2,
        k: int = 5
    ):
        super().__init__(doc_retriever, code_retriever, doc_weight, code_weight, k)
        self.keyword_retriever = keyword_retriever
        self.keyword_weight = keyword_weight

    def get_relevant_documents(self, query: str) -> List[Document]:
        """三路检索：文档 + 代码 + 关键词"""
        if not self.code_enabled and self.keyword_retriever is None:
            return self.doc_retriever.get_relevant_documents(query)

        logger.info(f"三路检索: {query[:50]}...")

        results = []

        # 1. 文档检索
        doc_results = self.doc_retriever.get_relevant_documents(query)
        for doc in doc_results:
            doc.metadata['_score'] = doc.metadata.get('score', 0.5) * self.doc_weight
        results.extend(doc_results)
        logger.info(f"文档检索: {len(doc_results)} 条")

        # 2. 代码检索
        if self.code_enabled:
            code_results = self.code_retriever.get_relevant_documents(query)
            for doc in code_results:
                doc.metadata['_score'] = doc.metadata.get('score', 0.5) * self.code_weight
            results.extend(code_results)
            logger.info(f"代码检索: {len(code_results)} 条")

        # 3. 关键词检索
        if self.keyword_retriever:
            try:
                keyword_results = self.keyword_retriever.get_relevant_documents(query)
                for doc in keyword_results:
                    doc.metadata['_score'] = doc.metadata.get('score', 0.3) * self.keyword_weight
                results.extend(keyword_results)
                logger.info(f"关键词检索: {len(keyword_results)} 条")
            except Exception as e:
                logger.warning(f"关键词检索失败: {e}")

        # 合并去重
        merged = self._merge_results(results, [])
        return merged[:self.k]
