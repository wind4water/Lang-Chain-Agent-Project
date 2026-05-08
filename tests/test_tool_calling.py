"""
测试工具调用功能
演示 Agent 如何自动调用工具回答问题
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.agents.sqlite_with_tools import SqliteAgentWithTools
from dotenv import load_dotenv

load_dotenv()


async def test_tools():
    """测试工具调用功能"""

    print("=" * 60)
    print("Tool Calling 功能测试")
    print("=" * 60)
    print()

    # 创建 Agent（启用工具）
    print("📦 正在初始化 Agent...")
    agent = SqliteAgentWithTools(
        db_path="checkpoints/test_tools.db",
        enable_tools=True
    )
    print()

    # 测试用例
    test_cases = [
        {
            "query": "现在几点了？",
            "description": "测试日期工具调用",
            "expected": "应该调用 get_current_date_tool"
        },
        {
            "query": "今天星期几？",
            "description": "测试星期查询工具",
            "expected": "应该调用 get_weekday"
        },
        {
            "query": "给我当前的时间戳",
            "description": "测试时间戳工具",
            "expected": "应该调用 get_current_timestamp"
        },
        {
            "query": "现在是几点？今天星期几？",
            "description": "测试多工具调用",
            "expected": "应该调用 get_current_date_tool 和 get_weekday"
        },
        {
            "query": "你好，请介绍一下自己",
            "description": "测试不需要工具的对话",
            "expected": "不应该调用工具，直接回答"
        },
    ]

    session_id = "tool_test_session"

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"测试 {i}/{len(test_cases)}: {case['description']}")
        print(f"{'=' * 60}")
        print(f"问题: {case['query']}")
        print(f"预期: {case['expected']}")
        print("-" * 60)

        try:
            response = await agent.chat(case['query'], session_id)
            print(f"\n✅ 回答:")
            print(response)
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}")

        print("\n" + "=" * 60)
        input("\n按 Enter 继续下一个测试...")

    # 查看历史
    print("\n" + "=" * 60)
    print("📝 查看完整对话历史")
    print("=" * 60)
    history = await agent.get_history(session_id)

    for i, msg in enumerate(history, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "user":
            print(f"\n{i}. 👤 用户: {content}")
        elif role == "assistant":
            print(f"\n{i}. 🤖 助手: {content}")
        elif role == "tool":
            tool_name = msg.get("tool_name", "unknown")
            print(f"\n{i}. 🔧 工具 ({tool_name}): {content}")

    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)


async def test_without_tools():
    """测试不启用工具的情况"""

    print("\n\n" + "=" * 60)
    print("测试：禁用工具的情况")
    print("=" * 60)
    print()

    agent = SqliteAgentWithTools(
        db_path="checkpoints/test_no_tools.db",
        enable_tools=False
    )

    session_id = "no_tools_session"

    # 同样的问题，但没有工具
    query = "现在几点了？"
    print(f"问题: {query}")
    print("-" * 60)

    try:
        response = await agent.chat(query, session_id)
        print(f"\n✅ 回答:")
        print(response)
        print("\n💡 注意: 因为没有工具，Agent 无法获取实时时间")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")

    print("\n" + "=" * 60)


async def main():
    """主函数"""

    print("""
╔══════════════════════════════════════════════════════════╗
║           LangChainProject Tool Calling 测试              ║
╚══════════════════════════════════════════════════════════╝

本测试将演示:
  ✅ Agent 自动判断是否需要工具
  ✅ 自动选择合适的工具
  ✅ 调用工具并获取结果
  ✅ 根据工具结果生成回答
  ✅ 对比启用/禁用工具的差异

""")

    print("选择测试模式:")
    print("1. 测试工具调用功能（启用工具）")
    print("2. 测试禁用工具的情况")
    print("3. 运行所有测试")
    print("0. 退出")

    choice = input("\n请输入选择 (0-3): ").strip()

    if choice == "1":
        await test_tools()
    elif choice == "2":
        await test_without_tools()
    elif choice == "3":
        await test_tools()
        await test_without_tools()
    elif choice == "0":
        print("再见！")
        return
    else:
        print("无效选择")


if __name__ == "__main__":
    asyncio.run(main())
