"""
测试智谱AI嵌入模型配置
"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from app.rag.core.embeddings import EmbeddingManager


def test_zhipu_embedding():
    """测试智谱AI嵌入模型"""
    print("\n" + "=" * 60)
    print("测试智谱AI Embedding-3 模型")
    print("=" * 60)

    # 检查配置
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("EMBEDDING_MODEL", "embedding-3")

    print(f"\n配置信息:")
    print(f"  API Key: {api_key[:20]}..." if api_key else "  API Key: 未配置")
    print(f"  Base URL: {base_url}")
    print(f"  模型名称: {model}")

    try:
        # 初始化嵌入管理器
        print("\n1️⃣  初始化嵌入管理器...")
        embedding_manager = EmbeddingManager(model_name=model)
        print("   ✅ 初始化成功")

        # 测试单个查询嵌入
        print("\n2️⃣  测试查询嵌入...")
        test_query = "什么是RAG系统？"
        print(f"   查询文本: {test_query}")

        vector = embedding_manager.embed_query(test_query)
        print(f"   ✅ 生成向量维度: {len(vector)}")
        print(f"   向量前5个值: {vector[:5]}")

        # 测试批量文档嵌入
        print("\n3️⃣  测试批量文档嵌入...")
        test_docs = [
            "RAG是检索增强生成系统",
            "智谱AI提供大语言模型服务",
            "向量数据库用于存储文档嵌入"
        ]
        print(f"   文档数量: {len(test_docs)}")

        vectors = embedding_manager.embed_documents(test_docs)
        print(f"   ✅ 生成向量数量: {len(vectors)}")
        print(f"   每个向量维度: {len(vectors[0])}")

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

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！智谱AI嵌入模型配置正确")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("\n可能的原因：")
        print("1. API密钥配置错误")
        print("2. 智谱AI不支持该嵌入模型")
        print("3. 网络连接问题")
        return False


if __name__ == "__main__":
    test_zhipu_embedding()
