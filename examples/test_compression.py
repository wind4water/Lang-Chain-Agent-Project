#!/usr/bin/env python3
"""
测试上下文压缩策略
"""
import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.sqlite import SqliteAgent


async def test_compression_strategy(strategy: str, rounds: int = 15):
    """测试指定的压缩策略"""
    print(f"\n{'='*60}")
    print(f"测试策略: {strategy.upper()}")
    print(f"{'='*60}\n")

    # 设置环境变量
    os.environ["CONTEXT_COMPRESSION_STRATEGY"] = strategy

    # 创建agent（使用测试专用的数据库）
    agent = SqliteAgent(db_path=f"checkpoints/test_{strategy}.db")

    session_id = f"compression-test-{strategy}"

    # 模拟多轮对话
    questions = [
        "你好，请介绍一下你自己",
        "Python的主要特点是什么？",
        "什么是面向对象编程？",
        "解释一下装饰器的概念",
        "列表和元组有什么区别？",
        "什么是生成器？",
        "如何处理异常？",
        "什么是上下文管理器？",
        "解释一下多线程和多进程",
        "什么是协程？",
        "FastAPI的优势是什么？",
        "什么是依赖注入？",
        "如何实现API认证？",
        "数据库连接池的作用是什么？",
        "什么是ORM？",
    ]

    for i in range(min(rounds, len(questions))):
        question = questions[i]
        print(f"\n[轮次 {i+1}] 用户: {question}")

        response = await agent.chat(question, session_id)
        print(f"[轮次 {i+1}] 助手: {response[:100]}...")

    # 获取最终统计
    print(f"\n{'='*60}")
    history = await agent.get_history(session_id)
    print(f"📊 最终统计:")
    print(f"   - 总消息数: {len(history)}")
    print(f"   - 总对话轮数: {len(history) // 2}")
    print(f"   - 预估tokens: {sum(len(msg['content']) for msg in history) // 4}")
    print(f"{'='*60}\n")


async def compare_all_strategies():
    """对比所有压缩策略"""
    strategies = ["none", "sliding_window", "token_limit", "summary"]

    print("🚀 开始对比测试所有压缩策略")
    print("测试配置: 15轮对话")

    for strategy in strategies:
        try:
            await test_compression_strategy(strategy, rounds=15)
            await asyncio.sleep(1)  # 避免API限流
        except Exception as e:
            print(f"❌ 策略 {strategy} 测试失败: {e}")


async def test_long_conversation():
    """测试超长对话场景"""
    print(f"\n{'='*60}")
    print("🔥 测试超长对话场景 (50轮)")
    print(f"{'='*60}\n")

    # 使用滑动窗口策略
    os.environ["CONTEXT_COMPRESSION_STRATEGY"] = "sliding_window"
    os.environ["COMPRESSION_WINDOW_SIZE"] = "10"

    agent = SqliteAgent(db_path="checkpoints/test_long.db")
    session_id = "long-conversation-test"

    # 模拟50轮对话
    for i in range(50):
        question = f"这是第{i+1}个问题，请简单回答"
        response = await agent.chat(question, session_id)
        print(f"[{i+1}/50] ✓")

    # 最终统计
    history = await agent.get_history(session_id)
    print(f"\n📊 超长对话测试结果:")
    print(f"   - 实际轮数: 50")
    print(f"   - 存储的消息数: {len(history)}")
    print(f"   - 压缩率: {len(history)/(50*2)*100:.1f}%")
    print(f"   - 预估tokens: {sum(len(msg['content']) for msg in history) // 4}")


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="测试上下文压缩策略")
    parser.add_argument(
        "--strategy",
        choices=["none", "sliding_window", "token_limit", "summary", "all"],
        default="all",
        help="测试的压缩策略"
    )
    parser.add_argument(
        "--long",
        action="store_true",
        help="测试超长对话场景（50轮）"
    )

    args = parser.parse_args()

    if args.long:
        await test_long_conversation()
    elif args.strategy == "all":
        await compare_all_strategies()
    else:
        await test_compression_strategy(args.strategy)

    print("\n✅ 测试完成！")


if __name__ == "__main__":
    asyncio.run(main())
