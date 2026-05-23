"""
RAG 系统管理器 - 多模型版本
支持路径级别的模型配置
"""
from typing import Optional, List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import os
import logging

from app.rag.config import rag_config
from app.rag.core.embeddings import EmbeddingManager
from app.rag.core.vectorstore import VectorStoreManager
from app.rag.core.hybrid_retriever import HybridRetriever
from app.rag.core.es_keyword_retriever import ESKeywordRetriever
from app.rag.core.chain import RAGChain
from app.rag.core.document_registry import DocumentRegistry
from app.rag.core.multi_model_manager import multi_model_manager
from app.rag.loaders.document_loader import DocumentLoader
from app.rag.processors.splitter import TextSplitter

logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG 系统管理器 - 支持多模型"""

    def __init__(self):
        """初始化 RAG 系统"""
        self.config = rag_config
        self.embedding_manager: Optional[EmbeddingManager] = None
        self.vectorstore_manager: Optional[VectorStoreManager] = None
        self.text_splitter: Optional[TextSplitter] = None
        self.rag_chain: Optional[RAGChain] = None
        self.llm: Optional[ChatOpenAI] = None
        self.document_registry: Optional[DocumentRegistry] = None
        self.es_keyword_retriever: Optional[ESKeywordRetriever] = None
        
        # 多模型支持
        self.multi_model_config: Optional[MultiModelConfig] = None
        self.model_managers: Dict[str, EmbeddingManager] = {}  # model_name -> EmbeddingManager
        self.model_vectorstores: Dict[str, VectorStoreManager] = {}  # model_name -> VectorStoreManager
        self._is_multi_model = bool(rag_config.multi_model_config)

        self._initialized = False

        logger.info("RAG 系统管理器已创建")
        if self._is_multi_model:
            logger.info("多模型模式已启用")

    def _rebuild_rag_chain(self):
        """基于当前向量库状态重建检索器与 RAG Chain。"""
        if self.vectorstore_manager is None or self.llm is None:
            logger.warning("跳过重建 RAG Chain：vectorstore_manager 或 llm 未初始化")
            return

        retriever = self._build_retriever()
        self.rag_chain = RAGChain(
            llm=self.llm,
            retriever=retriever
        )
        logger.info(
            "   ✅ 检索器已刷新: mode=%s, search_type=%s, Top-K=%s",
            self.config.retrieval_mode,
            self.config.search_type,
            self.config.top_k
        )

    def _build_retriever(self, metadata_filter: Optional[Dict[str, Any]] = None):
        """按当前配置构建检索器（支持多模型）。"""
        retrieval_mode = (self.config.retrieval_mode or "vector").lower()
        
        # 多模型模式：创建多路检索器
        if multi_model_manager.multi_model_config and len(self.model_vectorstores) > 1:
            logger.info("使用多模型检索器...")
            
            # 获取各模型的向量存储
            bge_retriever = None
            codebert_retriever = None
            
            for model_key, model_vs in self.model_vectorstores.items():
                model_config = multi_model_manager.multi_model_config.get_unique_models().get(
                    model_key.replace("bge_", "").replace("codebert_", "").replace("_", "/", 1)
                )
                
                # 创建检索器
                retriever = model_vs.get_retriever(
                    search_type=self.config.search_type,
                    k=self.config.top_k * 4,  # 每路多取一些，给 RRF 更充分的选择
                    score_threshold=self.config.score_threshold,
                    metadata_filter=metadata_filter
                )
                
                # 根据模型类型分配
                if model_config:
                    if model_config.model_type == "bge":
                        bge_retriever = retriever
                        logger.info(f"  BGE 检索器: {model_config.model_name}")
                    elif model_config.model_type == "codebert":
                        codebert_retriever = retriever
                        logger.info(f"  CodeBERT 检索器: {model_config.model_name}")
                else:
                    # 从 key 推断
                    if "bge" in model_key.lower():
                        bge_retriever = retriever
                        logger.info(f"  BGE 检索器 (推断): {model_key}")
                    else:
                        codebert_retriever = retriever
                        logger.info(f"  CodeBERT 检索器 (推断): {model_key}")
            
            # Hybrid 模式：三路 RRF 融合 (BGE + CodeBERT + ES)
            if retrieval_mode == "hybrid":
                from app.rag.core.multi_model_hybrid_retriever import MultiModelHybridRetriever
                logger.info("使用 RRF 三路融合检索器 (BGE + CodeBERT + ES)")
                return MultiModelHybridRetriever(
                    bge_retriever=bge_retriever,
                    codebert_retriever=codebert_retriever,
                    es_retriever=self.es_keyword_retriever,
                    rrf_k=60,  # RRF 常数，Google 推荐值
                    k=self.config.top_k,
                    per_retriever_k=self.config.top_k * 4  # 每路取更多，RRF 后取 top_k
                )
            
            # 非 hybrid 模式：使用原有的 MultiModelRetriever (BGE + CodeBERT 内部融合)
            from app.rag.core.multi_model_retriever import MultiModelRetriever
            
            retrievers = []
            total_weight = 0
            
            for model_key, model_vs in self.model_vectorstores.items():
                model_config = multi_model_manager.multi_model_config.get_unique_models().get(
                    model_key.replace("bge_", "").replace("codebert_", "").replace("_", "/", 1)
                )
                if model_config is None:
                    if "bge" in model_key.lower():
                        weight = self.config.doc_model_weight
                    else:
                        weight = self.config.code_model_weight
                else:
                    weight = self.config.doc_model_weight if model_config.model_type == "bge" else self.config.code_model_weight
                
                retriever = model_vs.get_retriever(
                    search_type=self.config.search_type,
                    k=self.config.top_k,
                    score_threshold=self.config.score_threshold,
                    metadata_filter=metadata_filter
                )
                
                retrievers.append({
                    "retriever": retriever,
                    "model_config": model_config,
                    "weight": weight
                })
                total_weight += weight
            
            # 归一化权重
            for r in retrievers:
                r["weight"] = r["weight"] / total_weight if total_weight > 0 else 1.0 / len(retrievers)
            
            return MultiModelRetriever(
                retrievers=retrievers,
                k=self.config.top_k
            )
        
        # 单模型模式：使用原有逻辑
        if retrieval_mode == "hybrid":
            return HybridRetriever(
                vectorstore_manager=self.vectorstore_manager,
                search_type=self.config.search_type,
                k=self.config.top_k,
                score_threshold=self.config.score_threshold,
                keyword_k=self.config.hybrid_keyword_k,
                metadata_filter=metadata_filter,
                es_keyword_retriever=self.es_keyword_retriever
            )
        if retrieval_mode != "vector":
            logger.warning("未知检索模式: %s，回退为 vector", retrieval_mode)

        return self.vectorstore_manager.get_retriever(
            search_type=self.config.search_type,
            k=self.config.top_k,
            score_threshold=self.config.score_threshold,
            metadata_filter=metadata_filter
        )

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
                model_name=self.config.embedding_model,
                embedding_type=self.config.embedding_type,
                device=self.config.embedding_device
            )

            # 显示 Embedding 配置信息
            emb_info = self.embedding_manager.get_info()
            logger.info(f"   ✅ 嵌入类型: {emb_info['type']}")
            logger.info(f"   ✅ 嵌入模型: {emb_info['model']}")
            logger.info(f"   ✅ 向量维度: {emb_info['dimension']}")
            if emb_info['type'] == 'local':
                logger.info(f"   ✅ 运行设备: {emb_info['device']}")
                logger.info(f"   💡 首次使用会自动下载模型（约 400MB），请耐心等待...")
            else:
                logger.info(f"   ✅ API 端点: {emb_info['api_endpoint']}")

            # 2. 初始化向量存储（支持多模型）
            logger.info("2️⃣  初始化向量存储...")
            
            if multi_model_manager.multi_model_config:
                # 多模型模式：初始化所有配置的模型
                logger.info("   🔄 使用多模型模式...")
                
                # 为每个模型创建向量库（如果不存在则创建）
                for model_name, model_config in multi_model_manager.multi_model_config.get_unique_models().items():
                    logger.info(f"   📚 初始化模型: {model_config.model_name}")
                    
                    # 获取或创建嵌入管理器
                    embedding_manager = multi_model_manager.get_or_create_embedding_manager(model_config)
                    
                    # 创建向量存储
                    from app.rag.core.vectorstore import VectorStoreManager
                    vector_store_path = multi_model_manager.get_vector_store_path(model_config)
                    # 使用唯一的 collection_name，包含模型类型和模型名
                    safe_model_name = model_config.model_name.replace("/", "_").replace("-", "_")
                    collection_name = f"{self.config.collection_name}_{model_config.model_type}_{safe_model_name}"
                    
                    model_vs = VectorStoreManager(
                        embeddings=embedding_manager.embeddings,
                        persist_directory=vector_store_path,
                        collection_name=collection_name
                    )
                    
                    self.model_vectorstores[f"{model_config.model_type}_{model_config.model_name}"] = model_vs
                    logger.info(f"   ✅ {model_config.model_name}: {os.path.abspath(vector_store_path)}")
                
                # 使用第一个模型作为主向量库（兼容旧代码）
                first_model = list(multi_model_manager.multi_model_config.path_configs)[0]
                first_key = f"{first_model.model_type}_{first_model.model_name}"
                self.vectorstore_manager = self.model_vectorstores[first_key]
                
                logger.info(f"   ✅ 主向量库: {first_model.model_name}")
            else:
                # 单模型模式：使用原有逻辑
                logger.info("   🔄 使用单模型模式...")
                vector_store_path = self.config.get_vector_store_path()
                self.vectorstore_manager = VectorStoreManager(
                    embeddings=self.embedding_manager.embeddings,
                    persist_directory=vector_store_path,
                    collection_name=self.config.collection_name
                )
                logger.info(f"   ✅ 存储路径: {os.path.abspath(vector_store_path)}")
                logger.info(f"   ℹ️  配置隔离: {self.config.embedding_type}_{self.config.embedding_model}")
            
            logger.info(f"   ✅ 向量数据库: Chroma")

            # 3. 初始化文本分割器
            logger.info("3️⃣  初始化文本分割器...")
            self.text_splitter = TextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            logger.info(f"   ✅ 块大小: {self.config.chunk_size}, 重叠: {self.config.chunk_overlap}")

            # 3.5 初始化 ES 关键词召回（可选）
            if self.config.es_enabled or self.config.keyword_backend.lower() == "es":
                logger.info("3️⃣.5️⃣  初始化 ES 关键词召回...")
                hosts = [h.strip() for h in self.config.es_hosts.split(",") if h.strip()]
                self.es_keyword_retriever = ESKeywordRetriever(
                    hosts=hosts,
                    index_name=self.config.es_index_name,
                    username=self.config.es_username,
                    password=self.config.es_password,
                    verify_certs=self.config.es_verify_certs,
                    ca_certs=self.config.es_ca_certs,
                    request_timeout=self.config.es_request_timeout,
                )
                if self.es_keyword_retriever.available:
                    logger.info("   ✅ ES 关键词召回已启用: %s", self.config.es_index_name)
                else:
                    logger.warning("   ⚠️ ES 不可用，将回退到本地关键词召回")

            # 4. 初始化文档注册表
            logger.info("4️⃣  初始化文档注册表...")
            # 根据 embedding 配置生成专属注册表路径
            registry_path = self.config.get_registry_path()
            self.document_registry = DocumentRegistry(registry_path)
            logger.info(f"   ✅ 注册表: {self.document_registry.get_stats()['total_documents']} 个文档")

            # 4.5 自动更新 Embedding 配置到注册表（首次或配置变化时）
            emb_info = self.embedding_manager.get_info()
            self.document_registry.update_embedding_config(
                embedding_type=emb_info['type'],
                model_name=emb_info['model'],
                dimension=emb_info['dimension']
            )

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
                model=os.getenv("MODEL_NAME", "glm-4.5-air"),
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL")
            )
            logger.info(f"   ✅ LLM: {self.llm.model_name}")

            # 7. 初始化 RAG Chain
            logger.info("7️⃣  初始化 RAG Chain...")
            self._rebuild_rag_chain()

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

        # 支持多个文档目录（逗号分隔）
        doc_paths = [p.strip() for p in documents_path.split(',')]
        existing_paths = [p for p in doc_paths if os.path.exists(p)]

        if not existing_paths:
            logger.warning(f"所有文档目录都不存在: {documents_path}")
            return {
                "status": "skipped",
                "message": "文档目录不存在",
                "has_changes": False,
                "added": 0,
                "modified": 0,
                "deleted": 0
            }

        try:
            # 1. 扫描目录，检测变更（支持多路径）
            changes = self.document_registry.scan_directory(
                documents_path,
                supported_extensions=set(DocumentLoader.get_supported_extensions())
            )

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
                # 尝试从所有路径中查找文件
                indexed_source = None
                for base_path in doc_paths:
                    potential_path = os.path.normpath(os.path.join(base_path, file_path))
                    if os.path.exists(potential_path):
                        indexed_source = potential_path
                        break

                if not indexed_source:
                    # 如果文件已删除，使用第一个路径尝试构建路径
                    indexed_source = os.path.normpath(os.path.join(doc_paths[0], file_path))

                # 多模型模式：从所有向量库删除（因为不知道原来在哪个库）
                if multi_model_manager.multi_model_config:
                    for model_key, model_vs in self.model_vectorstores.items():
                        try:
                            model_vs.delete_by_source(indexed_source)
                        except Exception as e:
                            logger.debug(f"从 {model_key} 删除失败（可能不存在）: {e}")
                else:
                    self.vectorstore_manager.delete_by_source(indexed_source)
                
                if self.es_keyword_retriever is not None and self.es_keyword_retriever.available:
                    self.es_keyword_retriever.delete_by_source(indexed_source)
                self.document_registry.unregister_document(file_path)

            # 3. 处理新增和修改的文档
            files_to_process = added_files + modified_files

            if files_to_process:
                # 加载文档（从多个路径中查找）
                documents = []

                for file_path in files_to_process:
                    # 查找文件在哪个基础路径下
                    found = False
                    for base_path in doc_paths:
                        full_path = os.path.join(base_path, file_path)
                        if os.path.exists(full_path):
                            try:
                                docs = DocumentLoader.load_file(full_path)
                                documents.extend(docs)
                                logger.debug(f"加载: {os.path.relpath(full_path, base_path)} ({len(docs)} 片段)")
                                found = True
                                break
                            except Exception as e:
                                logger.error(f"加载文档失败 {full_path}: {e}")

                    if not found:
                        logger.warning(f"文件未找到: {file_path}")

                if documents:
                    # 分割文档
                    split_docs = self.text_splitter.split_documents(documents)
                    logger.info(f"分割为 {len(split_docs)} 个文本块")

                    # 按来源分组
                    docs_by_source = {}
                    for doc in split_docs:
                        source = doc.metadata.get("source", "unknown")
                        # 计算相对路径（基于第一个匹配的基础路径）
                        rel_source = None
                        base_path_for_source = None
                        for base_path in doc_paths:
                            try:
                                rel = os.path.relpath(source, base_path)
                                if not rel.startswith('..'):  # 确保文件在此路径下
                                    rel_source = rel
                                    base_path_for_source = base_path
                                    break
                            except ValueError:
                                continue

                        if rel_source is None:
                            # 回退：使用第一个路径
                            rel_source = os.path.relpath(source, doc_paths[0])
                            base_path_for_source = doc_paths[0]

                        if rel_source not in docs_by_source:
                            docs_by_source[rel_source] = ([], base_path_for_source)
                        docs_by_source[rel_source][0].append(doc)

                    # 逐个文档更新向量存储（支持多模型）
                    for rel_source, (docs, base_path) in docs_by_source.items():
                        abs_source = os.path.join(base_path, rel_source)
                        indexed_source = os.path.normpath(abs_source)

                        if multi_model_manager.multi_model_config:
                            # 多模型模式：根据文件路径选择对应的向量库
                            model_config = multi_model_manager.get_model_for_file(abs_source)
                            model_key = f"{model_config.model_type}_{model_config.model_name}"
                            
                            if model_key in self.model_vectorstores:
                                model_vs = self.model_vectorstores[model_key]
                            else:
                                # 回退到主向量库
                                model_vs = self.vectorstore_manager
                                logger.warning(f"未找到模型 {model_key} 的向量库，使用主向量库")
                        else:
                            # 单模型模式
                            model_vs = self.vectorstore_manager

                        if rel_source in modified_files:
                            logger.info(f"🔄 更新: {rel_source}")
                            model_vs.update_documents(docs, indexed_source)
                            if self.es_keyword_retriever is not None and self.es_keyword_retriever.available:
                                self.es_keyword_retriever.update_documents(docs, indexed_source)
                        else:
                            logger.info(f"➕ 新增: {rel_source}")
                            model_vs.add_documents(docs)
                            if self.es_keyword_retriever is not None and self.es_keyword_retriever.available:
                                self.es_keyword_retriever.index_documents(docs)

                        # 更新注册表
                        metadata = self.document_registry.get_file_metadata(
                            abs_source,
                            base_path,
                            chunk_count=len(docs)
                        )
                        self.document_registry.register_document(metadata)

            # 4. 保存注册表
            self.document_registry.save()

            # 4.5 如果有变更，更新 Embedding 配置
            if total_changes > 0:
                emb_info = self.embedding_manager.get_info()
                self.document_registry.update_embedding_config(
                    embedding_type=emb_info['type'],
                    model_name=emb_info['model'],
                    dimension=emb_info['dimension']
                )
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

    async def _rebuild_multi_model(self, documents: List[Document]) -> int:
        """
        多模型重建知识库
        
        按模型配置将文档分组，分别索引到不同的向量库
        
        Args:
            documents: 文档列表
            
        Returns:
            索引的文档总数
        """
        logger.info("使用多模型模式重建...")
        
        # 1. 按模型分组文档
        docs_by_model = multi_model_manager.group_documents_by_model(documents)
        logger.info(f"文档分组: {len(docs_by_model)} 个模型")
        
        total_indexed = 0
        
        # 2. 为每个模型创建/获取向量库并索引
        for model_key, docs in docs_by_model.items():
            if not docs:
                continue
            
            # 获取模型配置
            sample_doc = docs[0]
            model_config = multi_model_manager.get_model_for_file(
                sample_doc.metadata.get("source", "")
            )
            
            logger.info(f"📚 索引到 {model_config.model_name}: {len(docs)} 个文档")
            
            try:
                # 获取或创建嵌入管理器
                embedding_manager = multi_model_manager.get_or_create_embedding_manager(model_config)
                
                # 获取向量存储路径
                persist_dir = multi_model_manager.get_vector_store_path(model_config)
                
                # 创建向量存储管理器
                from app.rag.core.vectorstore import VectorStoreManager
                safe_model_name = model_config.model_name.replace("/", "_").replace("-", "_")
                collection_name = f"{self.config.collection_name}_{model_config.model_type}_{safe_model_name}"
                vectorstore = VectorStoreManager(
                    embeddings=embedding_manager.embeddings,
                    persist_directory=persist_dir,
                    collection_name=collection_name
                )
                
                # 清空并重建
                vectorstore.clear()
                ids = vectorstore.add_documents(docs)
                total_indexed += len(ids)
                
                logger.info(f"   ✅ {model_config.model_name}: {len(ids)} 个文档")
                
            except Exception as e:
                logger.error(f"索引到 {model_config.model_name} 失败: {e}")
                continue
        
        logger.info(f"✅ 多模型索引完成: {total_indexed} 个文档")
        return total_indexed

    async def rebuild_knowledge_base(self) -> Dict[str, Any]:
        """
        重建知识库（全量重建）

        Returns:
            重建结果统计
        """
        logger.info("开始重建知识库（全量）...")

        documents_path = self.config.documents_path

        # 支持多个文档目录（逗号分隔）
        doc_paths = [p.strip() for p in documents_path.split(',')]

        # 确保至少有一个目录存在
        existing_paths = []
        for path in doc_paths:
            if not os.path.exists(path):
                logger.warning(f"文档目录不存在: {path}，创建空目录")
                os.makedirs(path, exist_ok=True)
            existing_paths.append(path)

        try:
            # 1. 加载文档（从多个目录）
            all_documents = []
            for doc_path in existing_paths:
                logger.info(f"📂 从目录加载文档: {os.path.abspath(doc_path)}")
                documents = DocumentLoader.load_directory(
                    directory_path=doc_path,
                    recursive=True,
                    show_progress=True
                )
                all_documents.extend(documents)
                logger.info(f"   ✅ 从 {doc_path} 加载了 {len(documents)} 个文档片段")

            if not all_documents:
                logger.warning("未找到任何文档")
                return {
                    "status": "skipped",
                    "message": "未找到任何文档",
                    "document_count": 0
                }

            logger.info(f"   ✅ 总共加载了 {len(all_documents)} 个文档片段")

            # 2. 文本分割
            logger.info("✂️  分割文档...")
            split_docs = self.text_splitter.split_documents(all_documents)
            logger.info(f"   ✅ 分割为 {len(split_docs)} 个文本块")

            # 3. 重建向量存储（支持多模型）
            logger.info("🔄 重建向量存储...")
            
            if multi_model_manager.multi_model_config:
                # 多模型模式：按模型分组索引
                indexed_count = await self._rebuild_multi_model(split_docs)
            else:
                # 单模型模式：使用原有逻辑
                ids = self.vectorstore_manager.rebuild(split_docs)
                indexed_count = len(ids)
            
            logger.info(f"   ✅ 成功索引 {indexed_count} 个文本块")

            # 3.5 重建 ES 关键词索引（可选）
            if self.es_keyword_retriever is not None and self.es_keyword_retriever.available:
                logger.info("🔄 重建 ES 关键词索引...")
                es_count = self.es_keyword_retriever.rebuild(split_docs)
                logger.info(f"   ✅ ES 成功索引 {es_count} 个文本块")

            # 4. 重建文档注册表
            logger.info("📝 重建文档注册表...")
            self.document_registry.clear()

            # 按来源分组统计
            docs_by_source = {}
            for doc in split_docs:
                source = doc.metadata.get("source", "unknown")
                # 查找文件属于哪个基础路径
                base_path = None
                for doc_path in doc_paths:
                    try:
                        rel = os.path.relpath(source, doc_path)
                        if not rel.startswith('..'):  # 确保文件在此路径下
                            base_path = doc_path
                            break
                    except ValueError:
                        continue

                if base_path is None:
                    base_path = doc_paths[0]  # 回退到第一个路径

                rel_source = os.path.relpath(source, base_path)
                key = (rel_source, base_path)
                if key not in docs_by_source:
                    docs_by_source[key] = 0
                docs_by_source[key] += 1

            # 注册每个文档
            for (rel_source, base_path), chunk_count in docs_by_source.items():
                abs_source = os.path.join(base_path, rel_source)
                metadata = self.document_registry.get_file_metadata(
                    abs_source,
                    base_path,
                    chunk_count=chunk_count
                )
                self.document_registry.register_document(metadata)

            self.document_registry.save()
            logger.info(f"   ✅ 注册了 {len(docs_by_source)} 个文档")

            # 4.5 更新 Embedding 配置
            if multi_model_manager.multi_model_config:
                # 多模型模式：使用第一个模型的信息
                first_model = list(multi_model_manager.multi_model_config.path_configs)[0]
                self.document_registry.update_embedding_config(
                    embedding_type=first_model.model_type,
                    model_name=first_model.model_name,
                    dimension=first_model.dimension
                )
            else:
                emb_info = self.embedding_manager.get_info()
                self.document_registry.update_embedding_config(
                    embedding_type=emb_info['type'],
                    model_name=emb_info['model'],
                    dimension=emb_info['dimension']
                )
            self.document_registry.save()

            result = {
                "status": "success",
                "original_documents": len(all_documents),
                "split_documents": len(split_docs),
                "indexed_documents": indexed_count,
                "registered_files": len(docs_by_source),
                "collection_name": self.config.collection_name
            }

            # 5. 重建后刷新 retriever，避免继续使用旧 collection 引用
            self._rebuild_rag_chain()

            logger.info("✅ 知识库重建完成")
            return result

        except Exception as e:
            logger.error(f"❌ 重建知识库失败: {e}")
            raise

    async def query(
        self,
        question: str,
        with_sources: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None,
        doc_group: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        RAG 查询

        Args:
            question: 用户问题
            with_sources: 是否返回来源信息
            metadata_filter: 元数据过滤条件（例如 {"filename": "xxx.md"}）
            doc_group: 可选的文档分组隔离（'documents' 或 'projects'）

        Returns:
            查询结果
        """
        if not self._initialized:
            raise RuntimeError("RAG 系统未初始化")

        # 1. 意图检测与检索范围收窄路由
        if not doc_group:
            # 方式 B: 基于轻量级关键词进行智能范围收窄
            q_lower = (question or "").lower()
            person_keywords = ["员工", "谁", "人", "同事", "部门", "口头禅", "岗位", "入职", "作者"]
            code_keywords = ["代码", "方法", "接口", "实现", "类", "enum", "枚举", "字段", "java", "xml", "class"]
            
            if any(k in q_lower for k in person_keywords):
                doc_group = "documents"
                logger.info(f"🧭 RAG 智能路由: 检测到员工/人相关的关键词，自动收窄检索范围至 doc_group='documents'")
                print(f"🧭 [RAG Router] 智能路由：检测到人物/人事相关意图，自动收窄检索范围至 doc_group='documents'")
            elif any(k in q_lower for k in code_keywords):
                doc_group = "projects"
                logger.info(f"🧭 RAG 智能路由: 检测到代码/技术相关的关键词，自动收窄检索范围至 doc_group='projects'")
                print(f"🧭 [RAG Router] 智能路由：检测到技术/代码相关意图，自动收窄检索范围至 doc_group='projects'")

        # 2. 如果存在 doc_group 约束，合并入 metadata_filter
        if doc_group:
            if not metadata_filter:
                metadata_filter = {}
            metadata_filter["doc_group"] = doc_group

        try:
            query_chain = self.rag_chain
            if metadata_filter:
                # 按请求构建带 metadata 过滤条件的 retriever，避免影响全局默认链路
                retriever = self._build_retriever(metadata_filter=metadata_filter)
                query_chain = RAGChain(
                    llm=self.llm,
                    retriever=retriever
                )

            if with_sources:
                return await query_chain.ainvoke_with_sources(question)
            else:
                answer = await query_chain.ainvoke(question)
                return {"answer": answer}
        except Exception as e:
            # 部分场景下（如重建后）旧 retriever 仍指向已删除 collection，自动刷新并重试一次
            if "collection not initialized" in str(e).lower():
                logger.warning("检测到 collection 未初始化，刷新 retriever 后重试一次")
                self._rebuild_rag_chain()
                query_chain = self.rag_chain
                if metadata_filter:
                    retriever = self._build_retriever(metadata_filter=metadata_filter)
                    query_chain = RAGChain(
                        llm=self.llm,
                        retriever=retriever
                    )
                if with_sources:
                    return await query_chain.ainvoke_with_sources(question)
                else:
                    answer = await query_chain.ainvoke(question)
                    return {"answer": answer}
            raise

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
                "retrieval_mode": self.config.retrieval_mode,
                "keyword_backend": self.config.keyword_backend,
                "search_type": self.config.search_type,
                "hybrid_keyword_k": self.config.hybrid_keyword_k,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "rebuild_on_startup": self.config.rebuild_on_startup,
                "es_enabled": self.config.es_enabled,
                "es_index_name": self.config.es_index_name,
                "es_available": bool(self.es_keyword_retriever and self.es_keyword_retriever.available),
            }
        }


# 全局 RAG 系统实例
rag_system = RAGSystem()
