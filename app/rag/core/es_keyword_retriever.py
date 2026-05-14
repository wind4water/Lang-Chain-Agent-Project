"""
Elasticsearch 关键词召回模块
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ESKeywordRetriever:
    """基于 Elasticsearch 的关键词召回器。"""

    def __init__(
        self,
        hosts: List[str],
        index_name: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        verify_certs: bool = True,
        ca_certs: Optional[str] = None,
        request_timeout: int = 30,
    ):
        self.hosts = hosts
        self.index_name = index_name
        self.username = username
        self.password = password
        self.verify_certs = verify_certs
        self.ca_certs = ca_certs
        self.request_timeout = request_timeout

        self._client = None
        self._helpers = None
        self._available = False

        self._initialize_client()

    @property
    def available(self) -> bool:
        return self._available

    def _initialize_client(self):
        """初始化 ES 客户端。失败时降级，不影响主流程。"""
        try:
            from elasticsearch import Elasticsearch
            from elasticsearch import helpers
        except Exception as e:
            logger.warning("ES 关键词召回不可用：未安装 elasticsearch 依赖 (%s)", e)
            self._available = False
            return

        kwargs: Dict[str, Any] = {
            "hosts": self.hosts,
            "verify_certs": self.verify_certs,
            "request_timeout": self.request_timeout,
        }
        if self.username:
            kwargs["basic_auth"] = (self.username, self.password or "")
        if self.ca_certs:
            kwargs["ca_certs"] = self.ca_certs

        try:
            client = Elasticsearch(**kwargs)
            if not client.ping():
                logger.warning("ES 连接失败：ping 不通 hosts=%s", self.hosts)
                self._available = False
                return

            self._client = client
            self._helpers = helpers
            self._available = True
            self.ensure_index()
            logger.info("✅ ES 关键词召回已启用，index=%s", self.index_name)
        except Exception as e:
            logger.warning("ES 初始化失败，关键词召回将降级为本地方案: %s", e)
            self._available = False

    @staticmethod
    def _doc_id(doc: Document) -> str:
        metadata = doc.metadata or {}
        source = str(metadata.get("source", ""))
        start_index = str(metadata.get("start_index", ""))
        content = doc.page_content or ""
        base = f"{source}|{start_index}|{content}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def _build_index_mapping(self) -> Dict[str, Any]:
        return {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard"
                        }
                    }
                }
            },
            "mappings": {
                "dynamic": True,
                "properties": {
                    "content": {"type": "text"},
                    "source": {"type": "keyword"},
                    "filename": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "extension": {"type": "keyword"},
                    "start_index": {"type": "integer"},
                    "metadata": {"type": "object", "enabled": True},
                }
            }
        }

    def ensure_index(self):
        if not self.available:
            return
        assert self._client is not None

        try:
            if not self._client.indices.exists(index=self.index_name):
                self._client.indices.create(
                    index=self.index_name,
                    body=self._build_index_mapping()
                )
                logger.info("已创建 ES 索引: %s", self.index_name)
        except Exception as e:
            logger.warning("确保 ES 索引失败: %s", e)

    def index_documents(self, documents: List[Document]) -> int:
        """批量写入（upsert）文档分块。"""
        if not self.available or not documents:
            return 0
        assert self._helpers is not None

        actions = []
        for doc in documents:
            metadata = doc.metadata or {}
            source = str(metadata.get("source", ""))
            filename = str(metadata.get("filename", ""))
            extension = str(metadata.get("extension", ""))
            start_index = metadata.get("start_index", 0)
            try:
                start_index = int(start_index)
            except Exception:
                start_index = 0

            actions.append({
                "_op_type": "index",
                "_index": self.index_name,
                "_id": self._doc_id(doc),
                "_source": {
                    "content": doc.page_content or "",
                    "source": source,
                    "filename": filename,
                    "extension": extension,
                    "start_index": start_index,
                    "metadata": metadata,
                }
            })

        if not actions:
            return 0

        try:
            success, _ = self._helpers.bulk(
                self._client,
                actions,
                stats_only=True,
                raise_on_error=False
            )
            logger.info("ES 已写入 %s 个文档块", success)
            return int(success)
        except Exception as e:
            logger.warning("ES 批量写入失败: %s", e)
            return 0

    def delete_by_source(self, source_path: str) -> int:
        """按 source 删除对应文档块。"""
        if not self.available:
            return 0
        assert self._client is not None

        try:
            result = self._client.delete_by_query(
                index=self.index_name,
                conflicts="proceed",
                query={"term": {"source": source_path}},
                refresh=True
            )
            deleted = int(result.get("deleted", 0))
            logger.info("ES 删除 source=%s, chunks=%s", source_path, deleted)
            return deleted
        except Exception as e:
            logger.warning("ES 删除 source 失败: %s", e)
            return 0

    def update_documents(self, documents: List[Document], source_path: str) -> int:
        """更新 source 对应文档（删除旧数据后写入新块）。"""
        self.delete_by_source(source_path)
        return self.index_documents(documents)

    def _build_filter_clauses(self, metadata_filter: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not metadata_filter:
            return []

        clauses: List[Dict[str, Any]] = []
        for key, value in metadata_filter.items():
            # 常用字段直接匹配；其余元数据走 metadata.<key>
            if key in {"source", "extension"}:
                clauses.append({"term": {key: value}})
            elif key == "filename":
                clauses.append({"term": {"filename.keyword": value}})
            else:
                clauses.append({"term": {f"metadata.{key}": value}})
        return clauses

    def search(
        self,
        query: str,
        k: int = 8,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """执行 ES 关键词召回。"""
        if not self.available or not query.strip():
            return []
        assert self._client is not None

        filter_clauses = self._build_filter_clauses(metadata_filter)
        body: Dict[str, Any] = {
            "size": k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content^2", "filename"],
                                "operator": "or"
                            }
                        }
                    ],
                    "filter": filter_clauses
                }
            }
        }

        try:
            result = self._client.search(index=self.index_name, body=body)
            hits = result.get("hits", {}).get("hits", [])
            docs: List[Document] = []
            for hit in hits:
                source = hit.get("_source", {})
                metadata = source.get("metadata", {}) or {}
                metadata.update({
                    "source": source.get("source", metadata.get("source", "")),
                    "filename": source.get("filename", metadata.get("filename", "")),
                    "extension": source.get("extension", metadata.get("extension", "")),
                    "start_index": source.get("start_index", metadata.get("start_index", 0)),
                    "_es_score": hit.get("_score", 0.0),
                })
                docs.append(Document(
                    page_content=source.get("content", ""),
                    metadata=metadata
                ))
            return docs
        except Exception as e:
            logger.warning("ES 关键词召回失败: %s", e)
            return []

    def clear(self):
        """清空 ES 索引并重建。"""
        if not self.available:
            return
        assert self._client is not None
        try:
            if self._client.indices.exists(index=self.index_name):
                self._client.indices.delete(index=self.index_name)
            self.ensure_index()
        except Exception as e:
            logger.warning("清空 ES 索引失败: %s", e)

    def rebuild(self, documents: List[Document]) -> int:
        """全量重建 ES 关键词索引。"""
        if not self.available:
            return 0
        self.clear()
        return self.index_documents(documents)

