"""
测试 RAG 智能同步功能
"""
import sys
import os
import asyncio
import shutil
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from app.rag import rag_system


async def test_smart_sync():
    """测试智能同步功能"""

    print("\n" + "=" * 70)
    print("RAG 智能同步功能测试")
    print("=" * 70)

    # 准备测试文档目录（使用唯一名称避免冲突）
    import time
    timestamp = int(time.time())
    test_docs_dir = f"data/documents_test_{timestamp}"
    test_chroma_dir = f"data/chroma_test_{timestamp}"

    os.makedirs(test_docs_dir, exist_ok=True)

    try:
        # 1. 准备新的 RAG 系统实例（避免全局单例冲突）
        print("\n1️⃣  创建新的 RAG 系统实例...")
        from app.rag.system import RAGSystem
        test_rag = RAGSystem()

        test_rag.config.documents_path = test_docs_dir
        test_rag.config.chroma_path = test_chroma_dir
        test_rag.config.collection_name = f"test_collection_{timestamp}"  # 使用唯一collection名
        test_rag.config.rebuild_on_startup = False
        test_rag.config.enabled = True

        await test_rag.initialize()
        print("   ✅ 初始化完成")

        # 2. 创建初始文档
        print("\n2️⃣  创建初始文档...")
        doc1 = Path(test_docs_dir) / "doc1.txt"
        doc2 = Path(test_docs_dir) / "doc2.txt"

        doc1.write_text("这是第一个文档。\n内容：Python 编程基础")
        doc2.write_text("这是第二个文档。\n内容：机器学习入门")
        print(f"   ✅ 创建了 2 个文档")

        # 3. 首次全量重建
        print("\n3️⃣  首次全量重建...")
        result = await test_rag.rebuild_knowledge_base()
        print(f"   ✅ 重建完成: {result}")

        # 4. 无变更同步（应该跳过）
        print("\n4️⃣  测试无变更同步...")
        result = await test_rag.sync_knowledge_base()
        print(f"   结果: has_changes={result['has_changes']}")
        assert not result["has_changes"], "应该无变更"
        print("   ✅ 正确跳过无变更同步")

        # 5. 新增文档
        print("\n5️⃣  新增文档...")
        doc3 = Path(test_docs_dir) / "doc3.txt"
        doc3.write_text("这是新增的第三个文档。\n内容：深度学习实战")
        print("   ✅ 新增了 doc3.txt")

        result = await test_rag.sync_knowledge_base()
        print(f"   同步结果: {result}")
        assert result["has_changes"], "应该检测到变更"
        assert result["added"] == 1, "应该新增1个文档"
        print("   ✅ 正确识别新增文档")

        # 6. 修改文档
        print("\n6️⃣  修改文档...")
        # 等待一点时间确保时间戳不同
        await asyncio.sleep(0.1)
        doc1.write_text("这是第一个文档（已修改）。\n内容：Python 高级编程")
        print("   ✅ 修改了 doc1.txt")

        result = await test_rag.sync_knowledge_base()
        print(f"   同步结果: {result}")
        assert result["has_changes"], "应该检测到变更"
        assert result["modified"] == 1, "应该修改1个文档"
        print("   ✅ 正确识别修改文档")

        # 7. 删除文档
        print("\n7️⃣  删除文档...")
        doc2.unlink()
        print("   ✅ 删除了 doc2.txt")

        result = await test_rag.sync_knowledge_base()
        print(f"   同步结果: {result}")
        assert result["has_changes"], "应该检测到变更"
        assert result["deleted"] == 1, "应该删除1个文档"
        print("   ✅ 正确识别删除文档")

        # 8. 混合操作
        print("\n8️⃣  混合操作（新增+修改+删除）...")
        await asyncio.sleep(0.1)

        # 新增 doc4
        doc4 = Path(test_docs_dir) / "doc4.txt"
        doc4.write_text("这是第四个文档。\n内容：自然语言处理")

        # 修改 doc3
        doc3.write_text("这是第三个文档（已修改）。\n内容：深度学习进阶")

        # 删除 doc1
        doc1.unlink()

        print("   ✅ 新增 doc4, 修改 doc3, 删除 doc1")

        result = await test_rag.sync_knowledge_base()
        print(f"   同步结果: {result}")
        assert result["has_changes"], "应该检测到变更"
        assert result["added"] == 1, "应该新增1个"
        assert result["modified"] == 1, "应该修改1个"
        assert result["deleted"] == 1, "应该删除1个"
        print("   ✅ 正确识别混合操作")

        # 9. 查看最终统计
        print("\n9️⃣  查看最终统计...")
        stats = test_rag.get_stats()
        print(f"   向量数据库文档块数: {stats['document_count']}")
        print(f"   注册表文档数: {stats['registry']['total_documents']}")
        print(f"   注册表文档块数: {stats['registry']['total_chunks']}")

        print("\n" + "=" * 70)
        print("✅ 所有测试通过！")
        print("=" * 70)

        print("\n💡 智能同步功能特点：")
        print("   - ✅ 自动检测文档变更（新增、修改、删除）")
        print("   - ✅ 只更新变更的部分，无需全量重建")
        print("   - ✅ 基于文件哈希精确判断变更")
        print("   - ✅ 支持混合操作一次性同步")
        print("   - ✅ 大幅提升更新速度")

    finally:
        # 清理测试数据
        print("\n🧹 清理测试数据...")
        if os.path.exists(test_docs_dir):
            shutil.rmtree(test_docs_dir)
        if os.path.exists(test_chroma_dir):
            shutil.rmtree(test_chroma_dir)
        print("   ✅ 清理完成")


if __name__ == "__main__":
    asyncio.run(test_smart_sync())
