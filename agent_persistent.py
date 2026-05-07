"""
升级版Agent - 支持持久化存储到SQLite数据库
上下文会保存到磁盘文件，重启服务后仍然存在
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import os
import sqlite3


class State(TypedDict):
    """对话状态定义"""
    messages: Annotated[list, add_messages]


class ConversationAgentWithPersistence:
    """支持持久化的对话Agent"""

    def __init__(self, db_path: str = "checkpoints/conversations.db"):
        # 检查API密钥
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("sk-your-api-key") or api_key == "your-api-key-here":
            raise ValueError("❌ 未配置API密钥！")

        # 初始化LLM
        self.llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            temperature=0.7,
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL")
        )

        # 创建Prompt模板
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="你是一个友好且乐于助人的AI助手。请用简洁、清晰的方式回答用户问题。"),
            MessagesPlaceholder(variable_name="messages"),
        ])

        # 创建Chain
        self.chain = self.prompt | self.llm | StrOutputParser()

        # 使用SqliteSaver - 持久化到磁盘
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # 创建数据库连接
        conn = sqlite3.connect(db_path, check_same_thread=False)

        # 使用同步版本的SqliteSaver
        self.checkpointer = SqliteSaver(conn)

        # 保存数据库路径
        self.db_path = db_path

        # 构建StateGraph
        self.graph = self._build_graph()

        print(f"✅ 使用持久化存储：{os.path.abspath(db_path)}")

    def _build_graph(self):
        """构建对话图"""
        workflow = StateGraph(State)
        workflow.add_node("chat", self._chat_node)
        workflow.add_edge(START, "chat")
        workflow.add_edge("chat", END)
        return workflow.compile(checkpointer=self.checkpointer)

    def _chat_node(self, state: State):
        """对话节点"""
        messages = state["messages"]
        response = self.chain.invoke({"messages": messages})
        return {"messages": [AIMessage(content=response)]}

    async def chat(self, message: str, session_id: str) -> str:
        """处理用户消息"""
        config = {"configurable": {"thread_id": session_id}}
        user_message = HumanMessage(content=message)
        result = await self.graph.ainvoke(
            {"messages": [user_message]},
            config=config
        )
        return result["messages"][-1].content

    async def get_history(self, session_id: str) -> list:
        """获取会话历史"""
        config = {"configurable": {"thread_id": session_id}}
        state = await self.graph.aget_state(config)

        if state and state.values.get("messages"):
            return [
                {
                    "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                    "content": msg.content
                }
                for msg in state.values["messages"]
            ]
        return []

    async def clear_history(self, session_id: str):
        """清除会话历史"""
        config = {"configurable": {"thread_id": session_id}}
        await self.graph.aupdate_state(config, {"messages": []})

    def list_all_sessions(self):
        """列出所有session_id（仅SQLite支持）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 查询checkpoints表中的所有thread_id
        try:
            cursor.execute("""
                SELECT DISTINCT json_extract(checkpoint, '$.configurable.thread_id') as thread_id
                FROM checkpoints
                WHERE thread_id IS NOT NULL
            """)
            sessions = [row[0] for row in cursor.fetchall()]
            conn.close()
            return sessions
        except Exception as e:
            conn.close()
            return []

    def get_database_stats(self):
        """获取数据库统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 获取总checkpoint数
            cursor.execute("SELECT COUNT(*) FROM checkpoints")
            total_checkpoints = cursor.fetchone()[0]

            # 获取唯一session数
            cursor.execute("""
                SELECT COUNT(DISTINCT json_extract(checkpoint, '$.configurable.thread_id'))
                FROM checkpoints
            """)
            total_sessions = cursor.fetchone()[0]

            # 获取数据库文件大小
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

            # 获取每个session的详细信息
            cursor.execute("""
                SELECT
                    json_extract(checkpoint, '$.configurable.thread_id') as thread_id,
                    COUNT(*) as checkpoint_count,
                    MIN(thread_ts) as first_seen,
                    MAX(thread_ts) as last_seen
                FROM checkpoints
                WHERE thread_id IS NOT NULL
                GROUP BY thread_id
                ORDER BY last_seen DESC
            """)

            sessions_detail = []
            for row in cursor.fetchall():
                sessions_detail.append({
                    "session_id": row[0],
                    "checkpoint_count": row[1],
                    "first_seen": row[2],
                    "last_seen": row[3]
                })

            conn.close()

            return {
                "total_sessions": total_sessions,
                "total_checkpoints": total_checkpoints,
                "database_size_bytes": db_size,
                "database_size_mb": round(db_size / 1024 / 1024, 2),
                "database_path": os.path.abspath(self.db_path),
                "sessions": sessions_detail
            }

        except Exception as e:
            conn.close()
            return {
                "error": str(e),
                "total_sessions": 0,
                "total_checkpoints": 0,
                "database_size_bytes": 0,
                "database_size_mb": 0,
                "database_path": os.path.abspath(self.db_path),
                "sessions": []
            }

    def get_session_detail(self, session_id: str):
        """获取指定session的详细信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 获取session的checkpoint信息
            cursor.execute("""
                SELECT
                    checkpoint_id,
                    parent_checkpoint_id,
                    thread_ts,
                    checkpoint
                FROM checkpoints
                WHERE json_extract(checkpoint, '$.configurable.thread_id') = ?
                ORDER BY thread_ts DESC
            """, (session_id,))

            checkpoints = []
            for row in cursor.fetchall():
                checkpoints.append({
                    "checkpoint_id": row[0],
                    "parent_checkpoint_id": row[1],
                    "timestamp": row[2],
                    "has_data": row[3] is not None
                })

            conn.close()

            return {
                "session_id": session_id,
                "checkpoint_count": len(checkpoints),
                "checkpoints": checkpoints
            }

        except Exception as e:
            conn.close()
            return {
                "session_id": session_id,
                "error": str(e),
                "checkpoint_count": 0,
                "checkpoints": []
            }
