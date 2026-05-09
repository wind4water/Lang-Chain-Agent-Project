#!/usr/bin/env python
"""
Langfuse 集成测试脚本（带详细日志）

测试 Langfuse 数据发送情况
"""

import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 设置 langfuse 日志为 DEBUG
logging.getLogger('langfuse').setLevel(logging.DEBUG)

# 加载环境变量
load_dotenv()

async def test_langfuse_integration():
    """测试 Langfuse 集成"""

    print("\n" + "="*60)
    print("Langfuse 集成测试（带详细日志）")
    print("="*60 + "\n")

    # 1. 检查配置
    print("1️⃣ 检查环境变量配置:")
    print("-" * 60)
    langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "false").lower()
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    print(f"   LANGFUSE_ENABLED: {langfuse_enabled}")
    print(f"   LANGFUSE_HOST: {host}")
    print(f"   LANGFUSE_PUBLIC_KEY: {public_key[:20] if public_key else '未设置'}...")
    print(f"   LANGFUSE_SECRET_KEY: {secret_key[:20] if secret_key else '未设置'}...")

    if langfuse_enabled != "true":
        print("\n⚠️  Langfuse 未启用！")
        print("   要启用 Langfuse，请在 .env 文件中设置:")
        print("   LANGFUSE_ENABLED=true")
        print("   LANGFUSE_PUBLIC_KEY=pk-lf-your-key")
        print("   LANGFUSE_SECRET_KEY=sk-lf-your-secret")
        return

    if not public_key or not secret_key:
        print("\n❌ Langfuse API 密钥未配置！")
        return

    if public_key.startswith("pk-lf-your-") or secret_key.startswith("sk-lf-your-"):
        print("\n⚠️  Langfuse 密钥为示例值，请更新为真实密钥！")
        return

    print("\n✅ 配置检查通过\n")

    # 2. 初始化 Agent
    print("2️⃣ 初始化 Agent:")
    print("-" * 60)

    from app.agents.sqlite_with_tools import SqliteAgentWithTools

    try:
        agent = SqliteAgentWithTools(enable_tools=False)  # 不启用工具，加快测试
        print(f"✅ Agent 初始化成功")
        print(f"   Langfuse 状态: {agent.langfuse_enabled}")
        print(f"   采样率: {getattr(agent, 'langfuse_sample_rate', 0.0)}")
    except Exception as e:
        print(f"❌ Agent 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # 3. 发送测试消息
    print("3️⃣ 发送测试消息:")
    print("-" * 60)

    test_message = "你好，这是一条测试消息"
    test_session = "langfuse-debug-test"

    try:
        print(f"📤 发送消息: {test_message}")
        print(f"📋 会话ID: {test_session}")
        print()

        response = await agent.chat(test_message, test_session)

        print()
        print(f"📥 收到回复: {response[:100] if len(response) > 100 else response}")
        print("✅ 对话成功")
    except Exception as e:
        print(f"❌ 对话失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # 4. 检查日志
    print("4️⃣ 日志检查:")
    print("-" * 60)
    print("请查看上面的日志输出，应该包含:")
    print("   🔵 创建 Langfuse callback: ...")
    print("   ✅ Langfuse callback 创建成功: ...")
    print("   🟢 Langfuse callback 已添加到 config")
    print("   📤 开始调用 LangGraph: ...")
    print("   📥 LangGraph 调用完成: ...")
    print("   🔄 开始 flush Langfuse 数据...")
    print("   ✅ Langfuse 数据 flush 完成")
    print()

    # 5. 验证建议
    print("5️⃣ 验证步骤:")
    print("-" * 60)
    print("1. 检查上面的日志是否有错误信息")
    print("2. 等待 5-10 秒让数据上传到 Langfuse")
    print("3. 访问 Langfuse 面板: https://cloud.langfuse.com")
    print("4. 进入你的项目 → Traces")
    print(f"5. 搜索 session_id: {test_session}")
    print("6. 应该能看到一条新的 trace 记录")
    print()

    print("="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_langfuse_integration())
