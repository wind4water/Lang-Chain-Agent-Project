"""
测试本地 Embedding 模型
"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from app.rag.core.embeddings import EmbeddingManager
import time


def test_local_embedding():
    """测试本地 Embedding 模型"""
    print("\n" + "=" * 60)
    print("测试本地 Embedding 模型")
    print("=" * 60)

    # 检查配置
    embedding_type = os.getenv("EMBEDDING_TYPE", "local")
    model = os.getenv("EMBEDDING_MODEL", "bge-base-zh-v1.5")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")

    print(f"\n配置信息:")
    print(f"  Embedding 类型: {embedding_type}")
    print(f"  模型名称: {model}")
    print(f"  运行设备: {device}")

    try:
        # 初始化嵌入管理器
        print("\n1️⃣  初始化本地嵌入管理器...")
        print("   💡 首次运行会自动下载模型（约 400MB），请耐心等待...")

        start_time = time.time()
        embedding_manager = EmbeddingManager(
            model_name=model,
            embedding_type="local",
            device=device
        )
        init_time = time.time() - start_time

        print(f"   ✅ 初始化成功 (耗时: {init_time:.2f}s)")

        # 获取配置信息
        info = embedding_manager.get_info()
        print(f"\n   模型信息:")
        print(f"   - 类型: {info['type']}")
        print(f"   - 模型: {info['model']}")
        print(f"   - 维度: {info['dimension']}")
        print(f"   - 设备: {info['device']}")

        # 测试单个查询嵌入
        print("\n2️⃣  测试查询嵌入...")
        test_query = "什么是RAG系统？"
        print(f"   查询文本: {test_query}")

        start_time = time.time()
        vector = embedding_manager.embed_query(test_query)
        query_time = time.time() - start_time

        print(f"   ✅ 生成向量维度: {len(vector)}")
        print(f"   ✅ 耗时: {query_time:.3f}s")
        print(f"   向量前5个值: {[f'{v:.4f}' for v in vector[:5]]}")

        # 测试批量文档嵌入
        print("\n3️⃣  测试批量文档嵌入...")
        test_docs = [
            "RAG是检索增强生成系统",
            "智谱AI提供大语言模型服务",
            "向量数据库用于存储文档嵌入",
            "LangChain是一个LLM应用开发框架",
            "Chroma是一个开源向量数据库"
        ]
        print(f"   文档数量: {len(test_docs)}")

        start_time = time.time()
        vectors = embedding_manager.embed_documents(test_docs)
        batch_time = time.time() - start_time

        print(f"   ✅ 生成向量数量: {len(vectors)}")
        print(f"   ✅ 每个向量维度: {len(vectors[0])}")
        print(f"   ✅ 总耗时: {batch_time:.3f}s")
        print(f"   ✅ 平均耗时: {batch_time/len(test_docs):.3f}s/doc")

        # 测试语义相似度
        print("\n4️⃣  测试语义相似度...")
        query_vec = embedding_manager.embed_query("RAG系统是什么？")
        doc1_vec = embedding_manager.embed_query("RAG是检索增强生成系统")
        doc2_vec = embedding_manager.embed_query("今天天气很好")

        # 计算余弦相似度
        import numpy as np

        def cosine_similarity(vec1, vec2):
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        sim1 = cosine_similarity(query_vec, doc1_vec)
        sim2 = cosine_similarity(query_vec, doc2_vec)

        print(f"   '什么是RAG系统' vs 'RAG是检索增强生成': {sim1:.4f}")
        print(f"   '什么是RAG系统' vs '今天天气很好': {sim2:.4f}")
        print(f"   ✅ 相关文本相似度更高: {sim1 > sim2}")

        # 性能测试
        print("\n5️⃣  性能测试...")
        test_text = "这是一段用于性能测试的文本" * 10
        iterations = 10

        start_time = time.time()
        for _ in range(iterations):
            embedding_manager.embed_query(test_text)
        total_time = time.time() - start_time

        print(f"   文本长度: {len(test_text)} 字符")
        print(f"   测试次数: {iterations}")
        print(f"   总耗时: {total_time:.2f}s")
        print(f"   平均耗时: {total_time/iterations:.3f}s/query")
        print(f"   吞吐量: {iterations/total_time:.2f} queries/s")

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！本地 Embedding 模型工作正常")
        print("=" * 60)

        # 提示信息
        print("\n💡 使用建议：")
        print("  1. 本地模型完全免费，无需 API Key")
        print("  2. 首次运行会下载模型，后续使用直接加载")
        print("  3. 模型缓存在: ~/.cache/huggingface/hub/")
        print("  4. 如有 GPU，设置 EMBEDDING_DEVICE=cuda 可大幅提速")
        print("  5. 批量处理性能更好，建议一次处理多个文档")

        return True

    except ImportError as e:
        print(f"\n❌ 依赖缺失: {e}")
        print("\n请安装必要的依赖：")
        print("  pip install langchain-huggingface sentence-transformers torch")
        return False
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_local_embedding()
    sys.exit(0 if success else 1)
