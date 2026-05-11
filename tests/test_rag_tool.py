"""
测试 RAG 工具集成

验证 RAG 工具能否被 Agent 正确调用
"""
import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from app.rag import rag_system
from app.agents.sqlite_with_tools import SqliteAgentWithTools


async def test_rag_tool_integration():
    """测试 RAG 工具集成"""

    print("\n" + "=" * 70)
    print("RAG 工具集成测试")
    print("=" * 70)

    # 1. 检查环境配置
    print("\n1️⃣  检查环境配置...")
    rag_enabled = os.getenv("RAG_ENABLED", "false").lower() == "true"
    print(f"   RAG_ENABLED: {rag_enabled}")

    if not rag_enabled:
        print("\n⚠️  RAG 未启用！")
        print("   请在 .env 文件中设置: RAG_ENABLED=true")
        print("   然后重新运行测试")
        return

    # 2. 初始化 RAG 系统
    print("\n2️⃣  初始化 RAG 系统...")
    try:
        await rag_system.initialize()
        print("   ✅ RAG 系统初始化成功")
    except Exception as e:
        print(f"   ❌ RAG 系统初始化失败: {e}")
        return

    # 3. 初始化 Agent
    print("\n3️⃣  初始化 Agent（启用工具）...")
    try:
        agent = SqliteAgentWithTools(enable_tools=True)
        print("   ✅ Agent 初始化成功")

        # 检查 RAG 工具是否加载
        tools = agent.list_available_tools()
        rag_tool_found = any(tool['name'] == 'knowledge_base_search' for tool in tools)

        if rag_tool_found:
            print("   ✅ RAG 工具已加载")
        else:
            print("   ❌ RAG 工具未找到")
            print(f"   已加载的工具: {[t['name'] for t in tools]}")
            return

    except Exception as e:
        print(f"   ❌ Agent 初始化失败: {e}")
        return

    # 4. 测试问题（应该触发 RAG 工具）
    test_questions = [
        "这个系统支持哪些文档格式？",
        "如何配置 RAG 功能？",
        "系统有哪些核心功能？"
    ]

    print("\n4️⃣  测试 Agent 自动调用 RAG 工具...")
    print("   （Agent 会根据问题自动决定是否使用知识库）\n")

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'─' * 70}")
        print(f"测试 {i}/{len(test_questions)}: {question}")
        print(f"{'─' * 70}")

        try:
            # 使用唯一的 session_id
            session_id = f"rag_tool_test_{i}"

            # 发送消息
            response = await agent.chat(question, session_id)

            print(f"\n📤 用户: {question}")
            print(f"🤖 Agent: {response}")

            # 清理测试会话
            await agent.clear_history(session_id)

        except Exception as e:
            print(f"   ❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ RAG 工具集成测试完成")
    print("=" * 70)

    print("\n💡 提示:")
    print("   - 如果 Agent 使用了 RAG 工具，回答中会包含「知识库回答」和「参考来源」")
    print("   - Agent 会根据问题内容自动决定是否调用知识库")
    print("   - 如果知识库中没有相关信息，Agent 可能会使用其他工具或直接回答")


async def test_rag_tool_directly():
    """直接测试 RAG 工具（不通过 Agent）"""

    print("\n" + "=" * 70)
    print("RAG 工具直接调用测试")
    print("=" * 70)

    # 初始化 RAG 系统
    print("\n1️⃣  初始化 RAG 系统...")
    try:
        await rag_system.initialize()
        print("   ✅ RAG 系统初始化成功")
    except Exception as e:
        print(f"   ❌ RAG 系统初始化失败: {e}")
        return

    # 直接调用 RAG 工具
    print("\n2️⃣  直接调用 RAG 工具...")
    try:
        from app.tools.rag_tool import RAGSearchTool

        rag_tool = RAGSearchTool()

        question = "系统支持哪些文档格式？"
        print(f"\n   问题: {question}")

        result = await rag_tool._arun(question)

        print(f"\n   结果:\n{result}")

        print("\n   ✅ RAG 工具直接调用成功")

    except Exception as e:
        print(f"   ❌ RAG 工具调用失败: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """主测试函数"""

    print("\n" + "=" * 70)
    print("🧪 RAG 工具集成完整测试套件")
    print("=" * 70)

    # 测试 1: 直接调用 RAG 工具
    await test_rag_tool_directly()

    print("\n\n")

    # 测试 2: Agent 集成测试
    await test_rag_tool_integration()

    print("\n\n" + "=" * 70)
    print("🎉 所有测试完成")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
