"""
测试启动时的 RAG 初始化行为
"""
import sys
import os
import asyncio
import logging

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

# 设置日志级别为 INFO
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

from app.rag import rag_system


async def test_startup():
    print("\n" + "=" * 70)
    print("测试 RAG 启动行为")
    print("=" * 70)

    print(f"\n配置:")
    print(f"  RAG_ENABLED = {rag_system.config.enabled}")
    print(f"  RAG_REBUILD_ON_STARTUP = {rag_system.config.rebuild_on_startup}")

    print(f"\n开始初始化...")
    await rag_system.initialize()

    print("\n" + "=" * 70)
    print("初始化完成！查看上面的日志：")
    print("  - 如果看到 '5️⃣  智能同步知识库（增量）' → 正确✅")
    print("  - 如果看到 '5️⃣  重建知识库（全量）' → 全量重建⚠️")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_startup())
