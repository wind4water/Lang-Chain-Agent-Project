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

            # 4. 如果需要，重建知识库
            if self.config.rebuild_on_startup:
                logger.info("4️⃣  重建知识库...")
                await self.rebuild_knowledge_base()
            else:
                logger.info("4️⃣  跳过知识库重建（配置为启动时不重建）")
                # 检查向量存储状态
                stats = self.vectorstore_manager.get_stats()
                logger.info(f"   ℹ️  当前文档数: {stats.get('document_count', 0)}")

            # 5. 初始化 LLM
            logger.info("5️⃣  初始化 LLM...")
            self.llm = ChatOpenAI(
                model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL")
            )
            logger.info(f"   ✅ LLM: {self.llm.model_name}")

            # 6. 初始化 RAG Chain
            logger.info("6️⃣  初始化 RAG Chain...")
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

    async def rebuild_knowledge_base(self) -> Dict[str, Any]:
        """
        重建知识库

        Returns:
            重建结果统计
        """
        logger.info("开始重建知识库...")

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

            result = {
                "status": "success",
                "original_documents": len(documents),
                "split_documents": len(split_docs),
                "indexed_documents": len(ids),
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

        return {
            "initialized": True,
            "enabled": self.config.enabled,
            "embedding_model": self.config.embedding_model,
            "vector_store": self.config.vector_store_type,
            "collection_name": self.config.collection_name,
            "document_count": vectorstore_stats.get("document_count", 0),
            "documents_path": os.path.abspath(self.config.documents_path),
            "vectordb_path": vectorstore_stats.get("persist_directory", ""),
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
