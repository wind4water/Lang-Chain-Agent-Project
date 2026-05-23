"""
代码嵌入模型管理 - 支持 UniXcoder 等代码专用模型
"""
from typing import Optional, List
from langchain_core.embeddings import Embeddings
import logging
import torch
import os

logger = logging.getLogger(__name__)


class UniXcoderEmbeddings(Embeddings):
    """
    UniXcoder 代码嵌入模型封装
    
    UniXcoder 是微软开源的代码预训练模型，专为代码理解和检索设计
    支持多种编程语言，在代码检索任务上比 CodeBERT 效果更好
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        初始化 UniXcoder 嵌入模型
        
        Args:
            model_name: HuggingFace 模型名称
            device: 设备类型 (cpu/cuda)，默认自动选择
            max_length: 最大序列长度
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self._tokenizer = None
        self._model = None
        self._load_model()
    
    def _load_model(self):
        """加载 UniXcoder 模型和分词器"""
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "使用 UniXcoder 需要安装 transformers\n"
                "请运行: pip install transformers torch"
            )
        
        logger.info(f"🔧 加载 UniXcoder 模型: {self.model_name}")
        
        # 加载分词器和模型
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=False
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=False
        )
        
        self._model.to(self.device)
        self._model.eval()
        
        logger.info(f"✅ UniXcoder 模型加载成功")
        logger.info(f"   - 模型: {self.model_name}")
        logger.info(f"   - 设备: {self.device}")
        logger.info(f"   - 维度: {self.get_dimension()}")
    
    def _encode_text(self, text: str) -> List[float]:
        """
        编码单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        # UniXcoder 使用特定的编码格式
        # 对于代码检索，使用 <encoder-only> 模式
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            # 使用 [CLS] token 的 embedding 作为句子表示
            embedding = outputs.last_hidden_state[:, 0, :]
            # 归一化
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy()[0].tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档
        
        Args:
            texts: 文档文本列表
            
        Returns:
            嵌入向量列表
        """
        logger.debug(f"Embedding {len(texts)} documents with UniXcoder")
        
        embeddings = []
        batch_size = 8  # 较小的批次以避免 OOM
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                try:
                    emb = self._encode_text(text)
                    batch_embeddings.append(emb)
                except Exception as e:
                    logger.error(f"Error encoding text: {e}")
                    # 返回零向量作为 fallback
                    batch_embeddings.append([0.0] * self.get_dimension())
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        嵌入查询文本
        
        Args:
            text: 查询文本
            
        Returns:
            嵌入向量
        """
        return self._encode_text(text)
    
    def get_dimension(self) -> int:
        """获取嵌入维度"""
        return 768  # UniXcoder-base 的维度


class CodeEmbeddingsFactory:
    """代码嵌入模型工厂"""
    
    @staticmethod
    def create_embeddings(
        model_type: str = "unixcoder",
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ) -> Embeddings:
        """
        创建代码嵌入模型
        
        Args:
            model_type: 模型类型 (unixcoder, codebert, etc.)
            model_name: 模型名称，默认根据类型选择
            device: 设备类型
            
        Returns:
            嵌入模型实例
        """
        if model_type.lower() == "unixcoder":
            model_name = model_name or "microsoft/unixcoder-base"
            return UniXcoderEmbeddings(
                model_name=model_name,
                device=device
            )
        elif model_type.lower() == "codebert":
            # 回退到原有的 HuggingFaceEmbeddings
            from langchain_huggingface import HuggingFaceEmbeddings
            model_name = model_name or "microsoft/codebert-base"
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device or 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            raise ValueError(f"不支持的代码嵌入模型类型: {model_type}")


def get_code_embeddings(
    model_name: str = "microsoft/unixcoder-base",
    device: Optional[str] = None
) -> Embeddings:
    """
    获取代码嵌入模型的便捷函数
    
    Args:
        model_name: 模型名称，默认 UniXcoder
        device: 设备类型
        
    Returns:
        代码嵌入模型实例
    """
    if "unixcoder" in model_name.lower():
        return UniXcoderEmbeddings(model_name=model_name, device=device)
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device or 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
