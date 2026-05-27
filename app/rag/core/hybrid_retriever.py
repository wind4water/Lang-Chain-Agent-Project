"""
Hybrid Retriever - 向量检索 + 关键词检索融合
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """混合检索器（向量 + 关键词）"""

    vectorstore_manager: Any
    search_type: str = "similarity"
    k: int = 4
    score_threshold: float = 0.5
    keyword_k: int = 8
    metadata_filter: Optional[Dict[str, Any]] = None
    es_keyword_retriever: Optional[Any] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
    VECTOR_WEIGHT: float = 2.0
    KEYWORD_WEIGHT: float = 1.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        轻量分词：英文词 + 单字中文。

        注：此方法仅用于 ES 不可用时的本地回退关键词匹配。
        生产环境中文分词由 ES IK 插件（ik_max_word / ik_smart）完成。
        """
        if not text:
            return []
        lowered = text.lower()
        return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lowered)

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """生成文档去重键。"""
        metadata = doc.metadata or {}
        source = metadata.get("source", "")
        filename = metadata.get("filename", "")
        start_index = metadata.get("start_index", "")
        preview = doc.page_content[:80] if doc.page_content else ""
        return f"{source}|{filename}|{start_index}|{preview}"

    @staticmethod
    def _doc_brief(doc: Document) -> str:
        metadata = doc.metadata or {}
        filename = metadata.get("filename") or metadata.get("source") or "unknown"
        start_index = metadata.get("start_index", "")
        return f"{filename}@{start_index}"

    def _keyword_search_local(self, query: str) -> List[Document]:
        """基于词项重叠的关键词召回。"""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            logger.info("Hybrid 本地关键词召回: query 为空或无有效 token")
            return []

        query_counter = Counter(query_tokens)
        collection = self.vectorstore_manager.vectorstore._collection

        get_kwargs: Dict[str, Any] = {"include": ["documents", "metadatas"]}
        if self.metadata_filter:
            get_kwargs["where"] = self.metadata_filter
        results = collection.get(**get_kwargs)

        docs = results.get("documents", []) or []
        metas = results.get("metadatas", []) or []
        if not docs:
            logger.info("Hybrid 本地关键词召回: 候选集为空")
            return []

        scored: List[tuple[float, Document]] = []
        for i, content in enumerate(docs):
            if not content:
                continue
            tokens = self._tokenize(content)
            if not tokens:
                continue
            token_counter = Counter(tokens)
            common = sum((query_counter & token_counter).values())
            if common == 0:
                continue
            # 兼顾匹配词数量与覆盖率
            score = float(common) + (common / max(len(query_counter), 1))
            metadata = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
            scored.append((score, Document(page_content=content, metadata=metadata)))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored[: self.keyword_k]]
        logger.info(
            "Hybrid 本地关键词召回完成: query=%s, candidates=%s, matched=%s, top_k=%s",
            query,
            len(docs),
            len(scored),
            len(top_docs),
        )
        return top_docs

    def _keyword_search(self, query: str) -> List[Document]:
        """关键词召回：优先 ES，失败时回退本地词项重叠。"""
        if self.es_keyword_retriever is not None and getattr(self.es_keyword_retriever, "available", False):
            logger.info("Hybrid 关键词召回后端: ES")
            docs = self.es_keyword_retriever.search(
                query=query,
                k=self.keyword_k,
                metadata_filter=self.metadata_filter
            )
            if docs:
                logger.info("Hybrid ES 关键词召回命中: %s", len(docs))
                return docs
            logger.info("Hybrid ES 关键词召回为空，回退本地关键词召回")
        else:
            logger.info("Hybrid 关键词召回后端: local（ES 不可用）")
        return self._keyword_search_local(query)

    def _vector_search(self, query: str) -> List[Document]:
        """向量检索。"""
        logger.info(
            "Hybrid 向量检索开始: search_type=%s, k=%s, score_threshold=%s, metadata_filter=%s",
            self.search_type,
            self.k,
            self.score_threshold,
            self.metadata_filter,
        )
        retriever = self.vectorstore_manager.get_retriever(
            search_type=self.search_type,
            k=self.k,
            score_threshold=self.score_threshold,
            metadata_filter=self.metadata_filter,
        )
        docs = retriever.invoke(query)
        logger.info("Hybrid 向量检索完成: query=%s, hits=%s", query, len(docs))
        return docs

    def _hybrid_search(self, query: str) -> List[Document]:
        logger.info("Hybrid 融合检索开始: query=%s", query)
        vector_docs = self._vector_search(query)
        keyword_docs = self._keyword_search(query)

        score_map: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # 向量召回优先级略高
        for i, doc in enumerate(vector_docs):
            key = self._doc_key(doc)
            doc_map[key] = doc
            score_map[key] = score_map.get(key, 0.0) + (
                self.VECTOR_WEIGHT * (len(vector_docs) - i)
            )

        for i, doc in enumerate(keyword_docs):
            key = self._doc_key(doc)
            doc_map[key] = doc
            score_map[key] = score_map.get(key, 0.0) + (
                self.KEYWORD_WEIGHT * float(len(keyword_docs) - i)
            )

        ranked_keys = sorted(score_map.keys(), key=lambda k: score_map[k], reverse=True)
        merged_docs = [doc_map[k] for k in ranked_keys[: self.k]]
        top_scores = [round(score_map[k], 3) for k in ranked_keys[: self.k]]
        top_docs_brief = [self._doc_brief(doc_map[k]) for k in ranked_keys[: self.k]]

        logger.info(
            "Hybrid 融合检索完成: vector_hits=%s, keyword_hits=%s, merged_hits=%s, "
            "vector_weight=%.2f, keyword_weight=%.2f, top_docs=%s, top_scores=%s",
            len(vector_docs),
            len(keyword_docs),
            len(merged_docs),
            self.VECTOR_WEIGHT,
            self.KEYWORD_WEIGHT,
            top_docs_brief,
            top_scores,
        )
        return merged_docs

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self._hybrid_search(query)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return await asyncio.to_thread(self._hybrid_search, query)

    @property
    def _type(self) -> str:
        """Type of retriever."""
        return "hybrid_retriever"
