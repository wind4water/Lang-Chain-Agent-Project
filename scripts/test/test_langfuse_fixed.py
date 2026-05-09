"""
测试 Langfuse 修复后的效果
"""
import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agents.sqlite_with_tools import SqliteAgentWithTools
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


async def test_chat_with_langfuse():
    """测试带 Langfuse 的对话"""
    print("=" * 60)
    print("测试 Langfuse 修复后的对话功能")
    print("=" * 60)

    # 创建 Agent（启用 Langfuse）
    agent = SqliteAgentWithTools(
        db_path="checkpoints/test_langfuse.db",
        enable_tools=True
    )

    # 测试会话
    session_id = "test_langfuse_fixed"

    try:
        # 测试 1: 简单对话
        print("\n📝 测试 1: 简单对话")
        response1 = await agent.chat("你好，请介绍一下你自己", session_id)
        print(f"回复: {response1[:100]}...")

        # 测试 2: 需要工具调用的对话
        print("\n📝 测试 2: 工具调用")
        response2 = await agent.chat("帮我搜索一下 Python 最新版本", session_id)
        print(f"回复: {response2[:100]}...")

        # 测试 3: 查看历史
        print("\n📝 测试 3: 查看历史")
        history = await agent.get_history(session_id)
        print(f"历史消息数: {len(history)}")

        print("\n✅ 所有测试完成！")
        print("\n💡 检查点:")
        print("  1. 控制台是否还有 'NoneType' 或 'run not found' 错误？")
        print("  2. 功能是否正常（对话、工具调用、历史）？")
        print("  3. Langfuse 是否成功记录追踪数据？")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理
        print("\n🧹 清理测试数据...")
        await agent.clear_history(session_id)


if __name__ == "__main__":
    asyncio.run(test_chat_with_langfuse())
