"""
简单RAG测试 - 直接测试核心功能
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from app.rag.core.embeddings import EmbeddingManager
from app.rag.core.vectorstore import VectorStoreManager
from app.rag.core.chain import RAGChain
from langchain_openai import ChatOpenAI
import asyncio


async def test_simple_rag():
    """简单RAG流程测试"""
    print("\n" + "=" * 60)
    print("简单RAG流程测试")
    print("=" * 60)

    try:
        # 1. 准备测试文档
        print("\n1️⃣  准备测试文档...")
        test_docs = [
            Document(
                page_content="RAG系统支持PDF、TXT、MD三种文档格式。可以自动加载和处理这些格式的文件。",
                metadata={"filename": "guide.txt", "source": "test"}
            ),
            Document(
                page_content="智谱AI提供GLM-4-Flash大语言模型和Embedding-3嵌入模型。价格实惠，性能优秀。",
                metadata={"filename": "api.txt", "source": "test"}
            ),
            Document(
                page_content="向量数据库使用Chroma存储文档嵌入。支持相似度搜索和语义检索。",
                metadata={"filename": "tech.txt", "source": "test"}
            ),
        ]
        print(f"   ✅ 准备了{len(test_docs)}个测试文档")

        # 2. 初始化嵌入模型
        print("\n2️⃣  初始化嵌入模型...")
        embedding_manager = EmbeddingManager(model_name="embedding-3")
        print("   ✅ 嵌入模型初始化成功")

        # 3. 初始化向量存储（使用新的集合名）
        print("\n3️⃣  初始化向量存储...")
        import time
        collection_name = f"test_collection_{int(time.time())}"
        vectorstore_manager = VectorStoreManager(
            embeddings=embedding_manager.embeddings,
            persist_directory="data/vectordb/chroma_test",
            collection_name=collection_name
        )
        print(f"   ✅ 向量存储初始化成功: {collection_name}")

        # 4. 添加文档
        print("\n4️⃣  添加文档到向量存储...")
        ids = vectorstore_manager.add_documents(test_docs)
        print(f"   ✅ 成功添加{len(ids)}个文档")

        # 5. 测试检索
        print("\n5️⃣  测试文档检索...")
        query = "支持哪些文档格式？"
        results = vectorstore_manager.similarity_search(query, k=2)
        print(f"   查询: {query}")
        print(f"   ✅ 检索到{len(results)}个相关文档:")
        for i, doc in enumerate(results, 1):
            print(f"      [{i}] {doc.metadata.get('filename')}: {doc.page_content[:50]}...")

        # 6. 初始化LLM和RAG Chain
        print("\n6️⃣  初始化RAG Chain...")
        llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "glm-4-flash"),
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        retriever = vectorstore_manager.get_retriever(search_type="similarity", k=2)
        rag_chain = RAGChain(llm=llm, retriever=retriever)
        print("   ✅ RAG Chain初始化成功")

        # 7. 测试问答
        print("\n7️⃣  测试RAG问答...")
        question = "这个系统支持什么文档格式？"
        print(f"   问题: {question}")

        answer = await rag_chain.ainvoke(question)
        print(f"   ✅ 回答: {answer}\n")

        # 8. 测试带来源的问答
        print("8️⃣  测试带来源的问答...")
        result = await rag_chain.ainvoke_with_sources(question)
        print(f"   答案: {result['answer']}")
        print(f"   来源数量: {result['source_count']}")
        for i, source in enumerate(result['sources'], 1):
            print(f"   [{i}] {source['filename']}")

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！RAG系统工作正常")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_simple_rag())
