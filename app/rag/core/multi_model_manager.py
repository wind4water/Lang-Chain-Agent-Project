"""
多模型 RAG 系统集成模块

在现有 RAGSystem 基础上增加多模型支持
通过在文档处理阶段分发到不同模型
"""
from typing import Dict, List, Optional
from langchain_core.documents import Document
import logging
import os

from app.rag.config import rag_config
from app.rag.core.multi_model_config import MultiModelConfig, PathModelConfig
from app.rag.core.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class MultiModelManager:
    """
    多模型管理器
    
    负责管理多个嵌入模型和对应的向量库
    """

    def __init__(self):
        self.config = rag_config
        self.multi_model_config: Optional[MultiModelConfig] = None
        self.embedding_managers: Dict[str, EmbeddingManager] = {}
        self._initialized = False

        # 如果配置了多模型，初始化配置
        if self.config.multi_model_config:
            self.multi_model_config = MultiModelConfig(self.config.multi_model_config)
            logger.info(f"多模型配置加载成功: {len(self.multi_model_config.path_configs)} 个模型")

    def should_use_multi_model(self, file_path: str) -> bool:
        """判断是否应该使用多模型处理该文件"""
        if not self.multi_model_config:
            return False
        # 检查是否有非默认的配置匹配
        model_config = self.multi_model_config.get_model_for_path(file_path)
        return model_config.path != "*"  # 不是默认配置

    def get_model_for_file(self, file_path: str) -> PathModelConfig:
        """获取文件对应的模型配置"""
        if self.multi_model_config:
            return self.multi_model_config.get_model_for_path(file_path)
        # 返回默认配置
        return PathModelConfig(
            path="*",
            model_type=self.config.embedding_type,
            model_name=self.config.embedding_model,
            dimension=self.config.embedding_dimension,
            description="默认模型"
        )

    def get_or_create_embedding_manager(self, model_config: PathModelConfig) -> EmbeddingManager:
        """
        获取或创建指定模型的 EmbeddingManager
        
        Args:
            model_config: 模型配置

        Returns:
            EmbeddingManager 实例
        """
        cache_key = f"{model_config.model_type}_{model_config.model_name}"
        
        if cache_key in self.embedding_managers:
            return self.embedding_managers[cache_key]

        logger.info(f"创建嵌入模型: {model_config.model_name}")
        
        try:
            # 根据模型类型创建对应的 EmbeddingManager
            if model_config.model_type == "bge":
                manager = EmbeddingManager(
                    model_name=model_config.model_name,
                    embedding_type="local",
                    device=self.config.embedding_device
                )
            elif model_config.model_type == "codebert":
                # CodeBERT 也使用 HuggingFace 加载
                manager = EmbeddingManager(
                    model_name=model_config.model_name,
                    embedding_type="local",
                    device=self.config.embedding_device
                )
            elif model_config.model_type == "openai":
                manager = EmbeddingManager(
                    model_name=model_config.model_name,
                    embedding_type="remote"
                )
            else:
                logger.warning(f"未知模型类型: {model_config.model_type}，使用默认 BGE")
                manager = EmbeddingManager(
                    model_name="BAAI/bge-base-zh-v1.5",
                    embedding_type="local"
                )

            self.embedding_managers[cache_key] = manager
            logger.info(f"✅ 嵌入模型创建成功: {model_config.model_name}")
            return manager

        except Exception as e:
            logger.error(f"创建嵌入模型失败: {e}")
            logger.warning("回退到默认 BGE 模型")
            manager = EmbeddingManager(
                model_name="BAAI/bge-base-zh-v1.5",
                embedding_type="local"
            )
            self.embedding_managers[cache_key] = manager
            return manager

    def get_vector_store_path(self, model_config: PathModelConfig) -> str:
        """生成向量存储路径"""
        safe_model_name = model_config.model_name.replace("/", "_").replace(":", "_")
        base_path = self.config.chroma_path.rstrip("/")
        return f"{base_path}/{model_config.model_type}_{safe_model_name}"

    def group_documents_by_model(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        按模型配置对文档分组
        
        Args:
            documents: 文档列表

        Returns:
            按 model_key 分组的文档字典
        """
        groups: Dict[str, List[Document]] = {}
        
        for doc in documents:
            source = doc.metadata.get("source", "")
            model_config = self.get_model_for_file(source)
            model_key = f"{model_config.model_type}_{model_config.model_name}"
            
            if model_key not in groups:
                groups[model_key] = []
            groups[model_key].append(doc)
        
        return groups

    def get_all_model_configs(self) -> List[PathModelConfig]:
        """获取所有唯一的模型配置"""
        if not self.multi_model_config:
            return [self.get_model_for_file("")]
        return list(self.multi_model_config.get_unique_models().values())


# 全局多模型管理器实例
multi_model_manager = MultiModelManager()
