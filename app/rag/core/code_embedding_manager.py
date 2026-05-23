"""
代码专用嵌入模型管理器
支持 CodeBERT 等代码嵌入模型
"""
from typing import List
from langchain_core.embeddings import Embeddings
import logging
import os

logger = logging.getLogger(__name__)


class CodeEmbeddingManager:
    """代码嵌入模型管理器"""

    def __init__(self, model_name: str = "microsoft/codebert-base", device: str = "cpu"):
        """
        初始化代码嵌入模型

        Args:
            model_name: 模型名称，默认 CodeBERT
            device: 运行设备 cpu/cuda
        """
        self.model_name = model_name
        self.device = device
        self._embeddings = None
        self._dimension = 768  # CodeBERT 默认维度

    @property
    def embeddings(self) -> Embeddings:
        """获取嵌入模型实例（懒加载）"""
        if self._embeddings is None:
            self._load_model()
        return self._embeddings

    def _load_model(self):
        """加载代码嵌入模型"""
        try:
            # 尝试使用 sentence-transformers
            from langchain_huggingface import HuggingFaceEmbeddings

            logger.info(f"加载代码嵌入模型: {self.model_name}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info(f"✅ 代码嵌入模型加载成功")
        except Exception as e:
            logger.error(f"加载代码嵌入模型失败: {e}")
            logger.info("尝试使用基础编码方式...")
            # 降级到普通编码
            self._embeddings = FallbackCodeEmbeddings()

    def get_info(self) -> dict:
        """获取模型信息"""
        return {
            "type": "code",
            "model": self.model_name,
            "dimension": self._dimension,
            "device": self.device
        }


class FallbackCodeEmbeddings(Embeddings):
    """代码嵌入降级方案（使用通用模型）"""

    def __init__(self):
        self._base_embeddings = None

    def _get_base(self):
        if self._base_embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._base_embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-zh-v1.5"
            )
        return self._base_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 对代码进行预处理，提取关键信息
        processed = [self._preprocess_code(t) for t in texts]
        return self._get_base().embed_documents(processed)

    def embed_query(self, text: str) -> List[float]:
        processed = self._preprocess_code(text)
        return self._get_base().embed_query(processed)

    def _preprocess_code(self, text: str) -> str:
        """
        预处理代码文本，提取关键信息
        """
        lines = text.split('\n')
        # 提取函数/类定义和注释
        keywords = []
        for line in lines:
            line = line.strip()
            # 提取 def/class 定义
            if line.startswith('def ') or line.startswith('class '):
                keywords.append(line)
            # 提取文档字符串
            elif '"""' in line or "'''" in line:
                keywords.append(line)
            # 提取关键注释
            elif line.startswith('#'):
                keywords.append(line[1:].strip())

        # 合并关键信息
        result = ' '.join(keywords[:10])  # 取前10个关键行
        if not result:
            result = text[:500]  # 降级到前500字符
        return result
