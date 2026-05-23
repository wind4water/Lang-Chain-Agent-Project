"""
多模型向量存储管理器
支持不同路径使用不同的嵌入模型和向量库
"""
from typing import Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from chromadb.config import Settings
import os
import logging

from app.rag.core.multi_model_config import MultiModelConfig, PathModelConfig
from app.rag.config import rag_config

logger = logging.getLogger(__name__)

# 全局 Chroma 客户端缓存，按 persist_dir 隔离
_chroma_client_cache: Dict[str, any] = {}


class MultiModelVectorStoreManager:
    """
    多模型向量存储管理器
    
    为不同路径维护独立的向量库和嵌入模型
    """

    def __init__(
        self,
        multi_model_config: Optional[MultiModelConfig] = None,
        base_persist_dir: str = "./data/vectordb/chroma"
    ):
        """
        初始化多模型管理器

        Args:
            multi_model_config: 多模型配置
            base_persist_dir: 向量库存储根目录
        """
        self.config = multi_model_config or MultiModelConfig()
        self.base_persist_dir = base_persist_dir
        
        # 模型实例缓存
        self._embeddings_cache: Dict[str, Embeddings] = {}
        # 向量库实例缓存
        self._vectorstore_cache: Dict[str, Chroma] = {}
        
        logger.info("多模型向量存储管理器已创建")

    def _get_embedding_model(self, model_config: PathModelConfig) -> Embeddings:
        """
        获取或创建嵌入模型实例

        Args:
            model_config: 模型配置

        Returns:
            嵌入模型实例
        """
        cache_key = f"{model_config.model_type}_{model_config.model_name}"
        
        if cache_key in self._embeddings_cache:
            return self._embeddings_cache[cache_key]

        logger.info(f"创建嵌入模型: {model_config.model_name}")
        
        try:
            if model_config.model_type == "bge":
                from langchain_huggingface import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_config.model_name,
                    model_kwargs={"device": rag_config.embedding_device},
                    encode_kwargs={"normalize_embeddings": True}
                )
            elif model_config.model_type == "codebert":
                from langchain_huggingface import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_config.model_name,
                    model_kwargs={"device": rag_config.embedding_device},
                    encode_kwargs={"normalize_embeddings": True}
                )
            elif model_config.model_type == "openai":
                from langchain_openai import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings(
                    model=model_config.model_name,
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    openai_api_base=os.getenv("OPENAI_BASE_URL")
                )
            else:
                raise ValueError(f"未知的模型类型: {model_config.model_type}")

            self._embeddings_cache[cache_key] = embeddings
            logger.info(f"✅ 嵌入模型创建成功: {model_config.model_name}")
            return embeddings

        except Exception as e:
            logger.error(f"创建嵌入模型失败: {e}")
            logger.warning("使用默认 BGE 模型")
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-zh-v1.5"
            )
            self._embeddings_cache[cache_key] = embeddings
            return embeddings

    def _get_vector_store_path(self, model_config: PathModelConfig) -> str:
        """生成向量库存储路径"""
        safe_model_name = model_config.model_name.replace("/", "_").replace(":", "_")
        return os.path.join(
            self.base_persist_dir,
            f"{model_config.model_type}_{safe_model_name}"
        )

    def _get_or_create_chroma_client(self, persist_dir: str):
        """获取或创建 Chroma 客户端（全局单例，按目录隔离）"""
        global _chroma_client_cache
        
        cache_key = os.path.abspath(persist_dir)
        
        if cache_key not in _chroma_client_cache:
            import chromadb
            logger.info(f"[MultiModel] 创建新的 Chroma 客户端: {cache_key}")
            _chroma_client_cache[cache_key] = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
        else:
            logger.info(f"[MultiModel] 复用已存在的 Chroma 客户端: {cache_key}")
        
        return _chroma_client_cache[cache_key]

    def get_vector_store_for_path(self, file_path: str) -> Chroma:
        """
        根据文件路径获取对应的向量库

        Args:
            file_path: 文件路径

        Returns:
            Chroma 向量库实例
        """
        model_config = self.config.get_model_for_path(file_path)
        cache_key = f"{model_config.model_type}_{model_config.model_name}"
        
        if cache_key in self._vectorstore_cache:
            return self._vectorstore_cache[cache_key]

        logger.info(f"为路径 {file_path} 创建向量库: {model_config.model_name}")
        
        embeddings = self._get_embedding_model(model_config)
        persist_dir = self._get_vector_store_path(model_config)
        
        # 使用唯一的 collection_name
        safe_model_name = model_config.model_name.replace("/", "_").replace("-", "_")
        collection_name = f"{rag_config.collection_name}_{model_config.model_type}_{safe_model_name}"
        
        # 使用全局缓存的客户端，避免重复创建
        client = self._get_or_create_chroma_client(persist_dir)
        
        vectorstore = Chroma(
            client=client,  # 关键：传入已创建的 PersistentClient
            collection_name=collection_name,
            embedding_function=embeddings,
            # 注意：传入 client 后不需要 persist_directory
        )
        
        self._vectorstore_cache[cache_key] = vectorstore
        logger.info(f"✅ 向量库创建成功: {persist_dir}")
        
        return vectorstore

    def add_documents(self, documents: List[Document]) -> Dict[str, List[str]]:
        """
        添加文档到对应的向量库
        
        根据文档的 source 路径，将文档添加到对应的模型向量库

        Args:
            documents: 文档列表

        Returns:
            按模型分类的文档 ID 字典
        """
        # 按路径分组
        docs_by_model: Dict[str, List[Document]] = {}
        
        for doc in documents:
            source = doc.metadata.get("source", "")
            model_config = self.config.get_model_for_path(source)
            model_key = f"{model_config.model_type}_{model_config.model_name}"
            
            if model_key not in docs_by_model:
                docs_by_model[model_key] = []
            docs_by_model[model_key].append(doc)
        
        # 分别添加到对应的向量库
        result_ids: Dict[str, List[str]] = {}
        for model_key, docs in docs_by_model.items():
            if not docs:
                continue
                
            # 获取该模型的向量库
            sample_doc = docs[0]
            vectorstore = self.get_vector_store_for_path(
                sample_doc.metadata.get("source", "")
            )
            
            logger.info(f"向 {model_key} 添加 {len(docs)} 个文档")
            
            # 分批添加（防止 Chroma 限制）
            batch_size = 2000
            ids = []
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                batch_ids = vectorstore.add_documents(batch)
                ids.extend(batch_ids)
            
            result_ids[model_key] = ids
            logger.info(f"✅ {model_key}: 成功添加 {len(ids)} 个文档")
        
        return result_ids

    def get_all_retrievers(self, search_type: str = "similarity", k: int = 4):
        """
        获取所有模型的检索器

        Args:
            search_type: 搜索类型
            k: 返回数量

        Returns:
            检索器列表
        """
        retrievers = []
        unique_models = self.config.get_unique_models()
        
        for model_name, model_config in unique_models.items():
            # 获取该模型的向量库
            cache_key = f"{model_config.model_type}_{model_config.model_name}"
            if cache_key in self._vectorstore_cache:
                vectorstore = self._vectorstore_cache[cache_key]
                retriever = vectorstore.as_retriever(
                    search_type=search_type,
                    search_kwargs={"k": k}
                )
                retrievers.append({
                    "retriever": retriever,
                    "model_config": model_config,
                    "weight": 1.0  # 可配置权重
                })
        
        return retrievers

    def get_stats(self) -> Dict[str, Dict]:
        """获取所有向量库的统计信息"""
        stats = {}
        for cache_key, vectorstore in self._vectorstore_cache.items():
            try:
                collection = vectorstore._collection
                count = collection.count()
                stats[cache_key] = {
                    "document_count": count,
                    "persist_directory": vectorstore._persist_directory
                }
            except Exception as e:
                stats[cache_key] = {"error": str(e)}
        return stats

    def clear_all(self):
        """清空所有向量库"""
        for cache_key, vectorstore in self._vectorstore_cache.items():
            try:
                vectorstore.delete_collection()
                logger.info(f"已清空向量库: {cache_key}")
            except Exception as e:
                logger.warning(f"清空向量库失败 {cache_key}: {e}")
        self._vectorstore_cache.clear()
