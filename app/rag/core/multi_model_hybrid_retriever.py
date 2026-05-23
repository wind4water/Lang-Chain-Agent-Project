"""
多模型 + ES 混合检索器
支持多模型向量检索结果与 ES 关键词检索结果融合
"""
from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
import logging

logger = logging.getLogger(__name__)


class MultiModelHybridRetriever(BaseRetriever):
    """
    多模型 Hybrid 检索器
    
    两阶段融合：
    1. 多模型向量检索（各模型内部已加权合并）
    2. 与 ES 关键词检索结果融合
    """
    
    # Pydantic v2 字段定义
    multi_retriever: Any = Field(..., description="多模型检索器")
    keyword_retriever: Optional[Any] = Field(None, description="ES关键词检索器")
    vector_weight: float = Field(0.7, description="向量权重")
    keyword_weight: float = Field(0.3, description="关键词权重")
    k: int = Field(5, description="返回结果数")
    
    class Config:
        arbitrary_types_allowed = True  # 允许任意类型

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        执行两阶段检索融合
        """
        # 阶段1：多模型向量检索
        vector_results = self.multi_retriever.get_relevant_documents(query)
        logger.info(f"多模型向量检索: {len(vector_results)} 条")

        # 给向量结果加权
        for doc in vector_results:
            base_score = doc.metadata.get("_weighted_score", 0.5)
            doc.metadata["_hybrid_score"] = base_score * self.vector_weight
            doc.metadata["_source_type"] = "vector"

        # 阶段2：ES 关键词检索（如果可用）
        keyword_results = []
        if self.keyword_retriever:
            try:
                keyword_results = self.keyword_retriever.get_relevant_documents(query)
                logger.info(f"ES 关键词检索: {len(keyword_results)} 条")
                
                # 给关键词结果加权
                for doc in keyword_results:
                    base_score = doc.metadata.get("score", 0.3)
                    doc.metadata["_hybrid_score"] = base_score * self.keyword_weight
                    doc.metadata["_source_type"] = "keyword"
            except Exception as e:
                logger.warning(f"ES 检索失败: {e}")

        # 合并两阶段结果
        all_results = list(vector_results) + list(keyword_results)

        # 去重
        unique_results = self._deduplicate(all_results)

        # 按融合分数排序
        unique_results.sort(
            key=lambda x: x.metadata.get("_hybrid_score", 0),
            reverse=True
        )

        logger.info(f"Hybrid 合并结果: {len(unique_results[:self.k])} 条")
        return unique_results[:self.k]

    def _deduplicate(self, documents: List[Document]) -> List[Document]:
        """基于 source 和内容去重"""
        seen = set()
        unique_docs = []

        for doc in documents:
            # 生成指纹
            source = doc.metadata.get("source", "")
            content_preview = doc.page_content[:150].strip()
            key = f"{source}_{hash(content_preview)}"

            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        return unique_docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """异步版本"""
        return self._get_relevant_documents(query)
