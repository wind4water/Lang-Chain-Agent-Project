"""
测试扫描 data/projects 目录
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.rag.loaders.document_loader import DocumentLoader
from app.rag.core.document_registry import DocumentRegistry
from app.rag.config import RAGConfig


def main():
    print("=" * 70)
    print("测试扫描 data/projects 目录")
    print("=" * 70)
    print()

    # 1. 检查配置
    config = RAGConfig()
    print(f"配置的文档路径: {config.documents_path}")
    print()

    # 2. 解析路径
    doc_paths = [p.strip() for p in config.documents_path.split(',')]
    print(f"解析后的路径:")
    for i, path in enumerate(doc_paths, 1):
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"  {i}. {path} {status}")
    print()

    # 3. 手动扫描 projects 目录
    projects_path = "./data/projects"
    if os.path.exists(projects_path):
        print(f"扫描 {projects_path}:")

        # 获取所有支持的扩展名
        extensions = set(DocumentLoader.get_supported_extensions())
        print(f"  支持的扩展名: {len(extensions)} 种")
        print()

        # 遍历目录
        found_files = []
        ignored_files = []

        for root, dirs, files in os.walk(projects_path):
            root_path = Path(root)

            # 检查目录是否应该忽略
            if DocumentLoader.should_ignore(root_path):
                print(f"  ❌ 忽略目录: {root}")
                # 移除这个目录，避免继续递归
                dirs.clear()
                continue

            # 检查文件
            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()

                # 检查扩展名
                if ext in extensions:
                    # 检查是否应该忽略
                    if DocumentLoader.should_ignore(file_path):
                        ignored_files.append(str(file_path))
                    else:
                        found_files.append(str(file_path))

        print(f"  找到 {len(found_files)} 个支持的文件")
        print(f"  忽略 {len(ignored_files)} 个文件")
        print()

        # 显示前 10 个找到的文件
        if found_files:
            print("  示例文件（前 10 个）:")
            for f in found_files[:10]:
                rel_path = os.path.relpath(f, projects_path)
                print(f"    ✅ {rel_path}")
        print()

    # 4. 使用 DocumentRegistry 扫描
    print("使用 DocumentRegistry 扫描:")
    registry_path = "./data/test_registry.json"
    registry = DocumentRegistry(registry_path)

    changes = registry.scan_directory(
        config.documents_path,
        supported_extensions=set(DocumentLoader.get_supported_extensions())
    )

    print(f"  新增: {len(changes['added'])} 个文件")
    print(f"  修改: {len(changes['modified'])} 个文件")
    print(f"  删除: {len(changes['deleted'])} 个文件")
    print()

    if changes['added']:
        print("  新增文件示例（前 10 个）:")
        for f in changes['added'][:10]:
            print(f"    ➕ {f}")

    # 清理测试注册表
    if os.path.exists(registry_path):
        os.remove(registry_path)

    print()
    print("=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
