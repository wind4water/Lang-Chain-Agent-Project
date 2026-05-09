"""
文档加载器
"""
from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader
)
import os
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """文档加载器"""

    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS = {
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader,
        '.pdf': PyPDFLoader,
    }

    @classmethod
    def load_file(cls, file_path: str) -> List[Document]:
        """
        加载单个文件

        Args:
            file_path: 文件路径

        Returns:
            文档列表
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        extension = file_path.suffix.lower()

        if extension not in cls.SUPPORTED_EXTENSIONS:
            logger.warning(f"不支持的文件格式: {extension}, 尝试作为文本文件加载")
            extension = '.txt'

        loader_class = cls.SUPPORTED_EXTENSIONS[extension]

        try:
            logger.info(f"加载文件: {file_path}")
            loader = loader_class(str(file_path))
            documents = loader.load()

            # 添加元数据
            for doc in documents:
                doc.metadata.update({
                    "source": str(file_path),
                    "filename": file_path.name,
                    "extension": extension
                })

            logger.info(f"✅ 成功加载 {len(documents)} 个文档片段")
            return documents

        except Exception as e:
            logger.error(f"加载文件失败: {file_path}, 错误: {e}")
            # 尝试作为纯文本加载
            try:
                logger.info(f"尝试作为纯文本加载: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                return [Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "extension": extension
                    }
                )]
            except Exception as e2:
                logger.error(f"纯文本加载也失败: {e2}")
                raise

    @classmethod
    def load_directory(
        cls,
        directory_path: str,
        glob_pattern: str = "**/*",
        recursive: bool = True,
        show_progress: bool = True
    ) -> List[Document]:
        """
        加载目录下的所有文档

        Args:
            directory_path: 目录路径
            glob_pattern: 文件匹配模式
            recursive: 是否递归子目录
            show_progress: 是否显示进度

        Returns:
            文档列表
        """
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")

        if not directory_path.is_dir():
            raise ValueError(f"不是目录: {directory_path}")

        logger.info(f"加载目录: {directory_path}")
        logger.info(f"匹配模式: {glob_pattern}, 递归: {recursive}")

        all_documents = []
        loaded_count = 0
        failed_count = 0

        # 遍历目录
        if recursive:
            files = list(directory_path.rglob(glob_pattern))
        else:
            files = list(directory_path.glob(glob_pattern))

        # 过滤出支持的文件
        supported_files = [
            f for f in files
            if f.is_file() and f.suffix.lower() in cls.SUPPORTED_EXTENSIONS
        ]

        logger.info(f"找到 {len(supported_files)} 个支持的文件")

        for file_path in supported_files:
            try:
                documents = cls.load_file(str(file_path))
                all_documents.extend(documents)
                loaded_count += 1

                if show_progress:
                    logger.info(f"进度: {loaded_count}/{len(supported_files)} - {file_path.name}")

            except Exception as e:
                failed_count += 1
                logger.error(f"加载文件失败: {file_path}, 错误: {e}")

        logger.info(f"✅ 目录加载完成:")
        logger.info(f"   - 成功: {loaded_count} 个文件")
        logger.info(f"   - 失败: {failed_count} 个文件")
        logger.info(f"   - 文档片段总数: {len(all_documents)}")

        return all_documents

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """
        获取支持的文件扩展名列表

        Returns:
            扩展名列表
        """
        return list(cls.SUPPORTED_EXTENSIONS.keys())
