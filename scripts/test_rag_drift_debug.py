import asyncio
import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from app.agents.sqlite_with_tools import SqliteAgentWithTools
from app.rag import rag_system

async def main():
    # 确保 RAG 系统已初始化
    await rag_system.initialize()
    
    agent = SqliteAgentWithTools(enable_tools=True)
    session_id = "test_debug_rag_drift"
    
    # 清理历史
    await agent.clear_history(session_id)
    
    print("\n--- 发送测试请求 ---\n")
    query = "员工许浩是谁？他经常说的话是什么"
    response = await agent.chat(query, session_id)
    
    print("\n--- 最终 Agent 回复 ---")
    print(response)
    print("---------------------\n")

if __name__ == "__main__":
    asyncio.run(main())
