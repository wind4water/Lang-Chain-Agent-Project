"""
文本分割处理器
"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


class TextSplitter:
    """文本分割器"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        """
        初始化文本分割器

        Args:
            chunk_size: 文本块大小
            chunk_overlap: 重叠大小
            separators: 分隔符列表
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 默认分隔符（优化中文分割）
        if separators is None:
            separators = [
                "\n\n",  # 段落
                "\n",    # 行
                "。",    # 中文句号
                "！",    # 中文感叹号
                "？",    # 中文问号
                "；",    # 中文分号
                ".",     # 英文句号
                "!",     # 英文感叹号
                "?",     # 英文问号
                ";",     # 英文分号
                " ",     # 空格
                "",      # 字符
            ]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

        logger.info(f"初始化文本分割器: chunk_size={chunk_size}, overlap={chunk_overlap}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档列表

        Args:
            documents: 文档列表

        Returns:
            分割后的文档列表
        """
        if not documents:
            logger.warning("没有文档需要分割")
            return []

        logger.info(f"开始分割 {len(documents)} 个文档")

        split_docs = self.splitter.split_documents(documents)

        logger.info(f"✅ 分割完成: {len(documents)} 个文档 → {len(split_docs)} 个文本块")

        return split_docs

    def split_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        分割单个文本

        Args:
            text: 文本内容
            metadata: 元数据

        Returns:
            分割后的文档列表
        """
        chunks = self.splitter.split_text(text)

        documents = [
            Document(
                page_content=chunk,
                metadata=metadata or {}
            )
            for chunk in chunks
        ]

        logger.info(f"文本分割: 1 个文本 → {len(documents)} 个文本块")

        return documents
