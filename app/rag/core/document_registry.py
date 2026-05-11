"""
文档注册表 - 跟踪已索引文档的元数据
"""
import os
import json
import hashlib
from typing import Dict, List, Set, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """文档元数据"""
    file_path: str  # 相对于documents_path的路径
    abs_path: str  # 绝对路径
    last_modified: float  # 文件最后修改时间
    file_hash: str  # 文件内容哈希
    file_size: int  # 文件大小
    chunk_count: int  # 分块数量


class DocumentRegistry:
    """
    文档注册表 - 管理已索引文档的元数据

    功能：
    - 记录已索引的文档信息（路径、修改时间、哈希值）
    - 检测文档变更（新增、修改、删除）
    - 支持增量更新
    """

    def __init__(self, registry_path: str):
        """
        初始化文档注册表

        Args:
            registry_path: 注册表文件路径（JSON格式）
        """
        self.registry_path = registry_path
        self.documents: Dict[str, DocumentMetadata] = {}

        # 确保目录存在
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)

        # 加载现有注册表
        self.load()

    def load(self):
        """从文件加载注册表"""
        if not os.path.exists(self.registry_path):
            logger.info("注册表文件不存在，创建新注册表")
            self.documents = {}
            return

        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.documents = {
                    k: DocumentMetadata(**v) for k, v in data.items()
                }
            logger.info(f"加载注册表: {len(self.documents)} 个文档")
        except Exception as e:
            logger.error(f"加载注册表失败: {e}")
            self.documents = {}

    def save(self):
        """保存注册表到文件"""
        try:
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                data = {
                    k: asdict(v) for k, v in self.documents.items()
                }
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"保存注册表: {len(self.documents)} 个文档")
        except Exception as e:
            logger.error(f"保存注册表失败: {e}")

    def compute_file_hash(self, file_path: str) -> str:
        """
        计算文件内容的哈希值

        Args:
            file_path: 文件路径

        Returns:
            SHA256哈希值
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # 分块读取，避免大文件占用过多内存
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希失败 {file_path}: {e}")
            return ""

    def get_file_metadata(self, file_path: str, base_path: str, chunk_count: int = 0) -> DocumentMetadata:
        """
        获取文件的元数据

        Args:
            file_path: 文件绝对路径
            base_path: 文档目录的基础路径
            chunk_count: 文档分块数量

        Returns:
            文档元数据
        """
        stat = os.stat(file_path)
        rel_path = os.path.relpath(file_path, base_path)

        return DocumentMetadata(
            file_path=rel_path,
            abs_path=os.path.abspath(file_path),
            last_modified=stat.st_mtime,
            file_hash=self.compute_file_hash(file_path),
            file_size=stat.st_size,
            chunk_count=chunk_count
        )

    def register_document(self, metadata: DocumentMetadata):
        """
        注册文档

        Args:
            metadata: 文档元数据
        """
        self.documents[metadata.file_path] = metadata

    def unregister_document(self, file_path: str):
        """
        取消注册文档

        Args:
            file_path: 文件相对路径
        """
        if file_path in self.documents:
            del self.documents[file_path]

    def is_registered(self, file_path: str) -> bool:
        """
        检查文档是否已注册

        Args:
            file_path: 文件相对路径

        Returns:
            是否已注册
        """
        return file_path in self.documents

    def has_changed(self, file_path: str, current_metadata: DocumentMetadata) -> bool:
        """
        检查文档是否已变更

        Args:
            file_path: 文件相对路径
            current_metadata: 当前文件的元数据

        Returns:
            是否已变更
        """
        if file_path not in self.documents:
            return True  # 新文件

        old_metadata = self.documents[file_path]

        # 先比较修改时间和大小（快速检查）
        if (old_metadata.last_modified != current_metadata.last_modified or
            old_metadata.file_size != current_metadata.file_size):
            return True

        # 如果时间和大小都没变，再比较哈希（慢速但准确）
        return old_metadata.file_hash != current_metadata.file_hash

    def scan_directory(
        self,
        documents_path: str,
        supported_extensions: Set[str] = {".txt", ".md", ".pdf"}
    ) -> Dict[str, List[str]]:
        """
        扫描文档目录，检测变更

        Args:
            documents_path: 文档目录路径
            supported_extensions: 支持的文件扩展名

        Returns:
            变更分类：{"added": [...], "modified": [...], "deleted": [...]}
        """
        if not os.path.exists(documents_path):
            logger.warning(f"文档目录不存在: {documents_path}")
            return {"added": [], "modified": [], "deleted": []}

        logger.info(f"扫描文档目录: {documents_path}")

        added = []
        modified = []
        current_files = set()

        # 遍历目录
        for root, _, files in os.walk(documents_path):
            for file in files:
                # 检查文件扩展名
                ext = os.path.splitext(file)[1].lower()
                if ext not in supported_extensions:
                    continue

                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, documents_path)
                current_files.add(rel_path)

                # 获取当前文件元数据（不计算哈希，快速扫描）
                try:
                    stat = os.stat(file_path)
                    current_metadata = DocumentMetadata(
                        file_path=rel_path,
                        abs_path=os.path.abspath(file_path),
                        last_modified=stat.st_mtime,
                        file_hash="",  # 暂不计算
                        file_size=stat.st_size,
                        chunk_count=0
                    )

                    # 检查是否是新文件或已修改
                    if not self.is_registered(rel_path):
                        added.append(rel_path)
                    else:
                        old_metadata = self.documents[rel_path]
                        # 快速检查（时间+大小）
                        if (old_metadata.last_modified != current_metadata.last_modified or
                            old_metadata.file_size != current_metadata.file_size):
                            # 需要计算哈希确认
                            current_metadata.file_hash = self.compute_file_hash(file_path)
                            if old_metadata.file_hash != current_metadata.file_hash:
                                modified.append(rel_path)

                except Exception as e:
                    logger.error(f"扫描文件失败 {file_path}: {e}")

        # 检查已删除的文件
        registered_files = set(self.documents.keys())
        deleted = list(registered_files - current_files)

        logger.info(f"扫描结果: 新增={len(added)}, 修改={len(modified)}, 删除={len(deleted)}")

        return {
            "added": added,
            "modified": modified,
            "deleted": deleted
        }

    def get_stats(self) -> Dict:
        """
        获取注册表统计信息

        Returns:
            统计信息
        """
        total_chunks = sum(doc.chunk_count for doc in self.documents.values())
        total_size = sum(doc.file_size for doc in self.documents.values())

        return {
            "total_documents": len(self.documents),
            "total_chunks": total_chunks,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "registry_path": self.registry_path
        }

    def clear(self):
        """清空注册表"""
        self.documents = {}
        self.save()
        logger.info("注册表已清空")
