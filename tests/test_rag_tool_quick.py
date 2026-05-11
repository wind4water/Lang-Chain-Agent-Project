"""
简化的 RAG 工具快速测试
"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

import asyncio
from app.rag import rag_system
from app.tools.rag_tool import RAGSearchTool


async def quick_test():
    print("\n" + "=" * 60)
    print("RAG 工具快速测试")
    print("=" * 60)

    # 1. 初始化 RAG 系统
    print("\n1️⃣  初始化 RAG 系统...")
    try:
        await rag_system.initialize()
        print("   ✅ RAG 系统初始化成功")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return

    # 2. 创建 RAG 工具
    print("\n2️⃣  创建 RAG 工具...")
    try:
        tool = RAGSearchTool()
        print(f"   ✅ 工具创建成功: {tool.name}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return

    # 3. 测试工具调用
    print("\n3️⃣  测试工具调用...")
    test_question = "系统支持哪些文档格式？"
    print(f"   问题: {test_question}")

    try:
        result = await tool._arun(test_question)
        print(f"\n   结果:\n{result}")
        print("\n   ✅ 工具调用成功")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(quick_test())
