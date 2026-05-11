"""
RAG 系统管理器
"""
from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
import os
import logging

from app.rag.config import rag_config
from app.rag.core.embeddings import EmbeddingManager
from app.rag.core.vectorstore import VectorStoreManager
from app.rag.core.chain import RAGChain
from app.rag.core.document_registry import DocumentRegistry
from app.rag.loaders.document_loader import DocumentLoader
from app.rag.processors.splitter import TextSplitter

logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG 系统管理器"""

    def __init__(self):
        """初始化 RAG 系统"""
        self.config = rag_config
        self.embedding_manager: Optional[EmbeddingManager] = None
        self.vectorstore_manager: Optional[VectorStoreManager] = None
        self.text_splitter: Optional[TextSplitter] = None
        self.rag_chain: Optional[RAGChain] = None
        self.llm: Optional[ChatOpenAI] = None
        self.document_registry: Optional[DocumentRegistry] = None

        self._initialized = False

        logger.info("RAG 系统管理器已创建")

    async def initialize(self):
        """初始化 RAG 系统"""
        if self._initialized:
            logger.info("RAG 系统已初始化，跳过")
            return

        if not self.config.enabled:
            logger.info("RAG 功能未启用")
            return

        logger.info("=" * 60)
        logger.info("初始化 RAG 系统...")
        logger.info("=" * 60)

        try:
            # 1. 初始化嵌入模型
            logger.info("1️⃣  初始化嵌入模型...")
            self.embedding_manager = EmbeddingManager(
                model_name=self.config.embedding_model
            )
            logger.info(f"   ✅ 嵌入模型: {self.config.embedding_model}")

            # 2. 初始化向量存储
            logger.info("2️⃣  初始化向量存储...")
            self.vectorstore_manager = VectorStoreManager(
                embeddings=self.embedding_manager.embeddings,
                persist_directory=self.config.chroma_path,
                collection_name=self.config.collection_name
            )
            logger.info(f"   ✅ 向量数据库: Chroma")
            logger.info(f"   ✅ 存储路径: {os.path.abspath(self.config.chroma_path)}")

            # 3. 初始化文本分割器
            logger.info("3️⃣  初始化文本分割器...")
            self.text_splitter = TextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            logger.info(f"   ✅ 块大小: {self.config.chunk_size}, 重叠: {self.config.chunk_overlap}")

            # 4. 初始化文档注册表
            logger.info("4️⃣  初始化文档注册表...")
            registry_path = os.path.join(self.config.chroma_path, "document_registry.json")
            self.document_registry = DocumentRegistry(registry_path)
            logger.info(f"   ✅ 注册表: {self.document_registry.get_stats()['total_documents']} 个文档")

            # 5. 智能同步知识库
            if self.config.rebuild_on_startup:
                logger.info("5️⃣  重建知识库（全量）...")
                await self.rebuild_knowledge_base()
            else:
                logger.info("5️⃣  智能同步知识库（增量）...")
                sync_result = await self.sync_knowledge_base()
                if sync_result["has_changes"]:
                    logger.info(f"   ✅ 同步完成: 新增={sync_result['added']}, "
                               f"修改={sync_result['modified']}, 删除={sync_result['deleted']}")
                else:
                    logger.info(f"   ℹ️  无变更，跳过同步")

                # 显示当前状态
                stats = self.vectorstore_manager.get_stats()
                logger.info(f"   ℹ️  当前文档数: {stats.get('document_count', 0)}")

            # 6. 初始化 LLM
            logger.info("6️⃣  初始化 LLM...")
            self.llm = ChatOpenAI(
                model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL")
            )
            logger.info(f"   ✅ LLM: {self.llm.model_name}")

            # 7. 初始化 RAG Chain
            logger.info("7️⃣  初始化 RAG Chain...")
            retriever = self.vectorstore_manager.get_retriever(
                search_type=self.config.search_type,
                k=self.config.top_k,
                score_threshold=self.config.score_threshold
            )
            self.rag_chain = RAGChain(
                llm=self.llm,
                retriever=retriever
            )
            logger.info(f"   ✅ 检索类型: {self.config.search_type}, Top-K: {self.config.top_k}")

            self._initialized = True

            logger.info("=" * 60)
            logger.info("✅ RAG 系统初始化完成")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"❌ RAG 系统初始化失败: {e}")
            raise

    async def sync_knowledge_base(self) -> Dict[str, Any]:
        """
        智能同步知识库（增量更新）

        扫描文档目录，检测变更，只更新变更的文档

        Returns:
            同步结果统计
        """
        logger.info("开始智能同步知识库...")

        documents_path = self.config.documents_path

        if not os.path.exists(documents_path):
            logger.warning(f"文档目录不存在: {documents_path}")
            return {
                "status": "skipped",
                "message": "文档目录不存在",
                "has_changes": False,
                "added": 0,
                "modified": 0,
                "deleted": 0
            }

        try:
            # 1. 扫描目录，检测变更
            changes = self.document_registry.scan_directory(documents_path)

            added_files = changes["added"]
            modified_files = changes["modified"]
            deleted_files = changes["deleted"]

            total_changes = len(added_files) + len(modified_files) + len(deleted_files)

            if total_changes == 0:
                logger.info("未检测到文档变更")
                return {
                    "status": "success",
                    "message": "无需同步",
                    "has_changes": False,
                    "added": 0,
                    "modified": 0,
                    "deleted": 0
                }

            logger.info(f"检测到 {total_changes} 个变更: "
                       f"新增={len(added_files)}, 修改={len(modified_files)}, 删除={len(deleted_files)}")

            # 2. 处理删除的文档
            for file_path in deleted_files:
                logger.info(f"❌ 删除: {file_path}")
                self.vectorstore_manager.delete_by_source(file_path)
                self.document_registry.unregister_document(file_path)

            # 3. 处理新增和修改的文档
            files_to_process = added_files + modified_files

            if files_to_process:
                # 加载文档
                full_paths = [os.path.join(documents_path, f) for f in files_to_process]
                documents = []

                for file_path in full_paths:
                    try:
                        docs = DocumentLoader.load_file(file_path)
                        documents.extend(docs)
                        logger.debug(f"加载: {os.path.relpath(file_path, documents_path)} ({len(docs)} 片段)")
                    except Exception as e:
                        logger.error(f"加载文档失败 {file_path}: {e}")

                if documents:
                    # 分割文档
                    split_docs = self.text_splitter.split_documents(documents)
                    logger.info(f"分割为 {len(split_docs)} 个文本块")

                    # 按来源分组
                    docs_by_source = {}
                    for doc in split_docs:
                        source = doc.metadata.get("source", "unknown")
                        rel_source = os.path.relpath(source, documents_path)
                        if rel_source not in docs_by_source:
                            docs_by_source[rel_source] = []
                        docs_by_source[rel_source].append(doc)

                    # 逐个文档更新向量存储
                    for rel_source, docs in docs_by_source.items():
                        abs_source = os.path.join(documents_path, rel_source)

                        if rel_source in modified_files:
                            logger.info(f"🔄 更新: {rel_source}")
                            self.vectorstore_manager.update_documents(docs, rel_source)
                        else:
                            logger.info(f"➕ 新增: {rel_source}")
                            self.vectorstore_manager.add_documents(docs)

                        # 更新注册表
                        metadata = self.document_registry.get_file_metadata(
                            abs_source,
                            documents_path,
                            chunk_count=len(docs)
                        )
                        self.document_registry.register_document(metadata)

            # 4. 保存注册表
            self.document_registry.save()

            result = {
                "status": "success",
                "has_changes": True,
                "added": len(added_files),
                "modified": len(modified_files),
                "deleted": len(deleted_files),
                "total_changes": total_changes
            }

            logger.info("✅ 智能同步完成")
            return result

        except Exception as e:
            logger.error(f"❌ 智能同步失败: {e}")
            raise

    async def rebuild_knowledge_base(self) -> Dict[str, Any]:
        """
        重建知识库（全量重建）

        Returns:
            重建结果统计
        """
        logger.info("开始重建知识库（全量）...")

        documents_path = self.config.documents_path

        if not os.path.exists(documents_path):
            logger.warning(f"文档目录不存在: {documents_path}，创建空目录")
            os.makedirs(documents_path, exist_ok=True)
            return {
                "status": "skipped",
                "message": "文档目录为空",
                "document_count": 0
            }

        try:
            # 1. 加载文档
            logger.info(f"📂 从目录加载文档: {os.path.abspath(documents_path)}")
            documents = DocumentLoader.load_directory(
                directory_path=documents_path,
                recursive=True,
                show_progress=True
            )

            if not documents:
                logger.warning("未找到任何文档")
                return {
                    "status": "skipped",
                    "message": "未找到任何文档",
                    "document_count": 0
                }

            logger.info(f"   ✅ 加载了 {len(documents)} 个文档片段")

            # 2. 文本分割
            logger.info("✂️  分割文档...")
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"   ✅ 分割为 {len(split_docs)} 个文本块")

            # 3. 重建向量存储
            logger.info("🔄 重建向量存储...")
            ids = self.vectorstore_manager.rebuild(split_docs)
            logger.info(f"   ✅ 成功索引 {len(ids)} 个文本块")

            # 4. 重建文档注册表
            logger.info("📝 重建文档注册表...")
            self.document_registry.clear()

            # 按来源分组统计
            docs_by_source = {}
            for doc in split_docs:
                source = doc.metadata.get("source", "unknown")
                rel_source = os.path.relpath(source, documents_path)
                if rel_source not in docs_by_source:
                    docs_by_source[rel_source] = 0
                docs_by_source[rel_source] += 1

            # 注册每个文档
            for rel_source, chunk_count in docs_by_source.items():
                abs_source = os.path.join(documents_path, rel_source)
                metadata = self.document_registry.get_file_metadata(
                    abs_source,
                    documents_path,
                    chunk_count=chunk_count
                )
                self.document_registry.register_document(metadata)

            self.document_registry.save()
            logger.info(f"   ✅ 注册了 {len(docs_by_source)} 个文档")

            result = {
                "status": "success",
                "original_documents": len(documents),
                "split_documents": len(split_docs),
                "indexed_documents": len(ids),
                "registered_files": len(docs_by_source),
                "collection_name": self.config.collection_name
            }

            logger.info("✅ 知识库重建完成")
            return result

        except Exception as e:
            logger.error(f"❌ 重建知识库失败: {e}")
            raise

    async def query(self, question: str, with_sources: bool = True) -> Dict[str, Any]:
        """
        RAG 查询

        Args:
            question: 用户问题
            with_sources: 是否返回来源信息

        Returns:
            查询结果
        """
        if not self._initialized:
            raise RuntimeError("RAG 系统未初始化")

        if with_sources:
            return await self.rag_chain.ainvoke_with_sources(question)
        else:
            answer = await self.rag_chain.ainvoke(question)
            return {"answer": answer}

    def get_stats(self) -> Dict[str, Any]:
        """
        获取 RAG 系统统计信息

        Returns:
            统计信息
        """
        if not self._initialized:
            return {
                "initialized": False,
                "enabled": self.config.enabled
            }

        vectorstore_stats = self.vectorstore_manager.get_stats()
        registry_stats = self.document_registry.get_stats()

        return {
            "initialized": True,
            "enabled": self.config.enabled,
            "embedding_model": self.config.embedding_model,
            "vector_store": self.config.vector_store_type,
            "collection_name": self.config.collection_name,
            "document_count": vectorstore_stats.get("document_count", 0),
            "documents_path": os.path.abspath(self.config.documents_path),
            "vectordb_path": vectorstore_stats.get("persist_directory", ""),
            "registry": registry_stats,
            "config": {
                "top_k": self.config.top_k,
                "search_type": self.config.search_type,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "rebuild_on_startup": self.config.rebuild_on_startup
            }
        }


# 全局 RAG 系统实例
rag_system = RAGSystem()
