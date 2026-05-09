"""
RAG 系统集成测试

测试 RAG（检索增强生成）系统的各项功能
"""
import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from app.rag import rag_system


async def test_rag_initialization():
    """测试 RAG 系统初始化"""
    print("\n" + "=" * 60)
    print("测试 1: RAG 系统初始化")
    print("=" * 60)

    try:
        await rag_system.initialize()
        print("✅ RAG 系统初始化成功")
        return True
    except Exception as e:
        print(f"❌ RAG 系统初始化失败: {e}")
        return False


async def test_rag_stats():
    """测试获取 RAG 统计信息"""
    print("\n" + "=" * 60)
    print("测试 2: 获取 RAG 统计信息")
    print("=" * 60)

    try:
        stats = rag_system.get_stats()
        print(f"✅ 获取统计信息成功:")
        print(f"   - 初始化状态: {stats.get('initialized')}")
        print(f"   - 启用状态: {stats.get('enabled')}")
        print(f"   - 文档数量: {stats.get('document_count')}")
        print(f"   - 嵌入模型: {stats.get('embedding_model')}")
        print(f"   - 向量存储: {stats.get('vector_store')}")
        return True
    except Exception as e:
        print(f"❌ 获取统计信息失败: {e}")
        return False


async def test_rebuild_knowledge_base():
    """测试重建知识库"""
    print("\n" + "=" * 60)
    print("测试 3: 重建知识库")
    print("=" * 60)

    try:
        result = await rag_system.rebuild_knowledge_base()
        print(f"✅ 知识库重建成功:")
        print(f"   - 状态: {result.get('status')}")
        print(f"   - 原始文档数: {result.get('original_documents')}")
        print(f"   - 分割文档数: {result.get('split_documents')}")
        print(f"   - 索引文档数: {result.get('indexed_documents')}")
        return True
    except Exception as e:
        print(f"❌ 知识库重建失败: {e}")
        return False


async def test_rag_query_simple():
    """测试简单 RAG 查询（不带来源）"""
    print("\n" + "=" * 60)
    print("测试 4: 简单 RAG 查询（不带来源）")
    print("=" * 60)

    question = "这个系统支持哪些工具？"
    print(f"问题: {question}")

    try:
        result = await rag_system.query(question, with_sources=False)
        print(f"✅ 查询成功")
        print(f"\n答案:\n{result.get('answer')}\n")
        return True
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        return False


async def test_rag_query_with_sources():
    """测试 RAG 查询（带来源）"""
    print("\n" + "=" * 60)
    print("测试 5: RAG 查询（带来源）")
    print("=" * 60)

    question = "如何配置 RAG 功能？"
    print(f"问题: {question}")

    try:
        result = await rag_system.query(question, with_sources=True)
        print(f"✅ 查询成功")
        print(f"\n答案:\n{result.get('answer')}\n")
        print(f"来源数量: {result.get('source_count')}")

        if result.get('sources'):
            print("\n来源详情:")
            for i, source in enumerate(result.get('sources', []), 1):
                print(f"\n  [{i}] {source.get('filename')}")
                print(f"      内容预览: {source.get('content_preview')[:100]}...")

        return True
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        return False


async def test_multiple_queries():
    """测试多个查询"""
    print("\n" + "=" * 60)
    print("测试 6: 批量查询测试")
    print("=" * 60)

    questions = [
        "这个系统的核心功能有哪些？",
        "如何降低 API 成本？",
        "支持哪些文档格式？",
        "如何部署到生产环境？",
    ]

    success_count = 0
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question}")
        try:
            result = await rag_system.query(question, with_sources=False)
            answer = result.get('answer', '')
            print(f"✅ 回答: {answer[:150]}...")
            success_count += 1
        except Exception as e:
            print(f"❌ 查询失败: {e}")

    print(f"\n批量查询完成: {success_count}/{len(questions)} 成功")
    return success_count == len(questions)


async def run_all_tests():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "RAG 系统集成测试" + " " * 27 + "║")
    print("╚" + "=" * 58 + "╝")

    # 检查 RAG 是否启用
    if not os.getenv("RAG_ENABLED", "false").lower() == "true":
        print("\n⚠️  RAG 功能未启用")
        print("   请在 .env 文件中设置: RAG_ENABLED=true")
        print("   然后重新运行测试")
        return

    # 检查文档目录
    docs_path = os.getenv("RAG_DOCUMENTS_PATH", "./data/documents")
    if not os.path.exists(docs_path) or not os.listdir(docs_path):
        print(f"\n⚠️  文档目录为空或不存在: {docs_path}")
        print("   请添加一些文档文件（.txt, .md, .pdf）到文档目录")
        return

    tests = [
        test_rag_initialization,
        test_rag_stats,
        test_rebuild_knowledge_base,
        test_rag_query_simple,
        test_rag_query_with_sources,
        test_multiple_queries,
    ]

    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ 测试执行异常: {e}")
            results.append(False)

    # 打印测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    print(f"失败: {total - passed}/{total}")

    if passed == total:
        print("\n✅ 所有测试通过！")
    else:
        print(f"\n⚠️  部分测试失败，请查看上方详细信息")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 运行测试
    asyncio.run(run_all_tests())
