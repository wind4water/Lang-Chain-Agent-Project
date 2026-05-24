"""
Reranker 模块 - 基于 Cross-Encoder 的重排序器

支持使用 BGE-Reranker-v2-m3 等本地模型对检索结果进行精排
"""
from typing import List, Optional
from langchain_core.documents import Document
import logging
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-Encoder 重排序器
    
    使用 Cross-Encoder 模型对 query-document 对进行相关性评分，
    比双塔模型的召回精度更高，适合作为精排阶段使用。
    
    Attributes:
        model_name: 模型名称或路径
        model: Cross-Encoder 模型
        tokenizer: 分词器
        device: 运行设备
        max_length: 最大序列长度
        batch_size: 批处理大小
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        max_length: int = 512,
        batch_size: int = 8
    ):
        """
        初始化 Cross-Encoder Reranker
        
        Args:
            model_name: 模型名称或本地路径，默认从环境变量 RERANKER_MODEL 获取
            device: 运行设备 ('cpu', 'cuda', 'mps')，默认自动选择
            max_length: 最大序列长度
            batch_size: 批处理大小
        """
        self.model_name = model_name or os.getenv("RERANKER_MODEL", "models/reranker/bge-reranker-v2-m3")
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        
        # 自动选择设备
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        self._load_model()
    
    def _load_model(self):
        """加载模型和分词器"""
        try:
            logger.info(f"加载 Reranker 模型: {self.model_name}")
            
            # 检查是否为本地路径
            if os.path.exists(self.model_name):
                model_path = self.model_name
                logger.info(f"使用本地模型: {model_path}")
            else:
                model_path = self.model_name
                logger.info(f"使用 HuggingFace 模型: {model_path}")
            
            # 加载分词器和模型
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True  # 离线模式
            )
            
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True  # 离线模式
            )
            
            self._model.to(self.device)
            self._model.eval()
            
            logger.info(f"✅ Reranker 模型加载完成，设备: {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Reranker 模型加载失败: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            top_k: 返回前 top_k 个结果，默认返回全部
            
        Returns:
            按相关性排序的文档列表
        """
        if not documents:
            return []
        
        if len(documents) == 1:
            # 单条结果直接返回
            documents[0].metadata['_rerank_score'] = 1.0
            return documents
        
        # 计算相关性分数
        scores = self._compute_scores(query, documents)
        
        # 添加分数到 metadata
        for doc, score in zip(documents, scores):
            doc.metadata['_rerank_score'] = float(score)
        
        # 按分数降序排序
        sorted_docs = sorted(
            documents,
            key=lambda x: x.metadata.get('_rerank_score', 0),
            reverse=True
        )
        
        # 返回前 top_k 个结果
        if top_k:
            sorted_docs = sorted_docs[:top_k]
        
        logger.info(f"Reranker 重排序完成: query='{query[:50]}...', {len(documents)} -> {len(sorted_docs)} 条")
        
        # 记录前几条结果的分数
        for i, doc in enumerate(sorted_docs[:3], 1):
            score = doc.metadata.get('_rerank_score', 0)
            filename = doc.metadata.get('filename', 'unknown')
            logger.info(f"  TOP-{i}: score={score:.4f}, file={filename}")
        
        return sorted_docs
    
    def _compute_scores(self, query: str, documents: List[Document]) -> List[float]:
        """
        计算 query-document 对的相关性分数
        
        Args:
            query: 查询文本
            documents: 文档列表
            
        Returns:
            相关性分数列表
        """
        all_scores = []
        
        # 批处理
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            
            # 构建 query-document 对
            pairs = []
            for doc in batch_docs:
                text = doc.page_content[:self.max_length * 2]  # 限制长度
                pairs.append([query, text])
            
            # 编码
            with torch.no_grad():
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # 前向传播
                outputs = self._model(**inputs)
                
                # 支持不同输出格式
                if outputs.logits.shape[-1] == 1:
                    # 单值回归输出 (BGE-Reranker-v2-m3)
                    batch_scores = torch.sigmoid(outputs.logits).squeeze(-1)
                else:
                    # 二分类输出
                    batch_scores = torch.softmax(outputs.logits, dim=-1)[:, 1]
                
                all_scores.extend(batch_scores.cpu().tolist())
        
        return all_scores
    
    def get_info(self) -> dict:
        """获取 Reranker 信息"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "loaded": self._model is not None
        }


class RerankerManager:
    """
    Reranker 管理器
    
    管理 Reranker 实例，支持懒加载和全局访问
    """
    
    _instance: Optional[CrossEncoderReranker] = None
    _enabled: bool = True
    
    @classmethod
    def get_reranker(cls) -> Optional[CrossEncoderReranker]:
        """获取 Reranker 实例（懒加载）"""
        if not cls._enabled:
            return None
        
        if cls._instance is None:
            try:
                # 检查环境变量是否禁用 reranker
                if os.getenv("RERANKER_ENABLED", "true").lower() == "false":
                    cls._enabled = False
                    logger.info("Reranker 已通过环境变量禁用")
                    return None
                
                cls._instance = CrossEncoderReranker()
                logger.info("✅ Reranker 懒加载完成")
            except Exception as e:
                logger.warning(f"⚠️ Reranker 初始化失败，将不使用重排序: {e}")
                cls._enabled = False
                return None
        
        return cls._instance
    
    @classmethod
    def is_enabled(cls) -> bool:
        """检查 Reranker 是否启用"""
        if not cls._enabled:
            return False
        return cls.get_reranker() is not None
    
    @classmethod
    def rerank(
        cls,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        对文档进行重排序（便捷方法）
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            top_k: 返回前 top_k 个结果
            
        Returns:
            按相关性排序的文档列表
        """
        reranker = cls.get_reranker()
        if reranker is None or not documents:
            # Reranker 未启用或文档为空，直接返回
            return documents
        
        return reranker.rerank(query, documents, top_k)
