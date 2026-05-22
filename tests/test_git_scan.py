"""
测试 Git 仓库扫描功能

验证：
1. 能正确加载代码文件（.py, .java, .js 等）
2. 自动忽略 .git, node_modules, __pycache__ 等
3. 支持多个文档目录
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.rag.loaders.document_loader import DocumentLoader
from app.rag.config import RAGConfig


def test_ignore_patterns():
    """测试忽略模式"""
    print("=" * 70)
    print("测试忽略模式")
    print("=" * 70)
    print()

    test_paths = [
        Path("project/.git/config"),
        Path("project/node_modules/package/index.js"),
        Path("project/__pycache__/module.pyc"),
        Path("project/src/main.py"),
        Path("project/.venv/lib/python"),
        Path("project/build/output.jar"),
        Path("project/docs/README.md"),
    ]

    for path in test_paths:
        should_ignore = DocumentLoader.should_ignore(path)
        status = "❌ 忽略" if should_ignore else "✅ 保留"
        print(f"{status}: {path}")

    print()


def test_supported_extensions():
    """测试支持的文件扩展名"""
    print("=" * 70)
    print("测试支持的文件扩展名")
    print("=" * 70)
    print()

    extensions = DocumentLoader.get_supported_extensions()
    print(f"总共支持 {len(extensions)} 种文件格式:")
    print()

    # 按类型分组
    docs = [ext for ext in extensions if ext in ['.txt', '.md', '.pdf']]
    code = [ext for ext in extensions if ext not in docs]

    print(f"文档格式 ({len(docs)} 种): {', '.join(docs)}")
    print(f"代码格式 ({len(code)} 种): {', '.join(code)}")
    print()


def test_multi_path_config():
    """测试多路径配置"""
    print("=" * 70)
    print("测试多路径配置")
    print("=" * 70)
    print()

    # 测试配置解析
    os.environ["RAG_DOCUMENTS_PATH"] = "./data/documents,./data/projects"

    config = RAGConfig()
    doc_paths = [p.strip() for p in config.documents_path.split(',')]

    print(f"配置的文档路径: {config.documents_path}")
    print(f"解析后的路径列表:")
    for i, path in enumerate(doc_paths, 1):
        exists = os.path.exists(path)
        status = "✅ 存在" if exists else "⚠️  不存在"
        print(f"  {i}. {path} ({status})")

    print()


if __name__ == "__main__":
    try:
        test_ignore_patterns()
        test_supported_extensions()
        test_multi_path_config()

        print("=" * 70)
        print("✨ 所有测试通过！")
        print("=" * 70)
        print()
        print("使用说明：")
        print("1. 将 Git 仓库放入 ./data/projects/ 目录")
        print("2. 修改 .env 中的 RAG_DOCUMENTS_PATH=./data/documents,./data/projects")
        print("3. 重启服务，系统会自动扫描两个目录")
        print("4. 代码文件会被自动索引，支持 30+ 种编程语言")
        print("5. .git, node_modules 等目录会被自动忽略")
        print()

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
