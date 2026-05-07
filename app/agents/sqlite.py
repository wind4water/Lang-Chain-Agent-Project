"""
升级版Agent - 支持持久化存储到SQLite数据库
上下文会保存到磁盘文件，重启服务后仍然存在
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import os
import aiosqlite


class State(TypedDict):
    """对话状态定义"""
    messages: Annotated[list, add_messages]


class SqliteAgent:
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

        # 保存数据库路径
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # checkpointer 将在首次使用时异步初始化
        self._conn = None
        self.checkpointer = None
        self.graph = None

        print(f"✅ 配置持久化存储：{os.path.abspath(db_path)}")

    async def _ensure_initialized(self):
        """确保checkpointer和graph已初始化"""
        if self.graph is None:
            # 直接创建 aiosqlite 连接
            self._conn = await aiosqlite.connect(self.db_path)

            # 使用连接创建 AsyncSqliteSaver
            self.checkpointer = AsyncSqliteSaver(self._conn)

            # 初始化数据库表（手动创建）
            await self._init_tables()

            # 构建StateGraph
            workflow = StateGraph(State)
            workflow.add_node("chat", self._chat_node)
            workflow.add_edge(START, "chat")
            workflow.add_edge("chat", END)

            self.graph = workflow.compile(checkpointer=self.checkpointer)
            print(f"✅ AsyncSqliteSaver初始化完成")

    async def _init_tables(self):
        """初始化数据库表"""
        cursor = await self._conn.cursor()

        # 创建 checkpoints 表
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint BLOB,
                metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
        """)

        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id
            ON checkpoints(thread_id)
        """)

        # 创建 checkpoint_writes 表
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoint_writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                type TEXT,
                value BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            )
        """)

        await self._conn.commit()
        print(f"✅ 数据库表已初始化")

    def _chat_node(self, state: State):
        """对话节点"""
        messages = state["messages"]
        response = self.chain.invoke({"messages": messages})
        return {"messages": [AIMessage(content=response)]}

    async def chat(self, message: str, session_id: str) -> str:
        """处理用户消息"""
        # 确保已初始化
        await self._ensure_initialized()

        config = {"configurable": {"thread_id": session_id}}
        user_message = HumanMessage(content=message)
        result = await self.graph.ainvoke(
            {"messages": [user_message]},
            config=config
        )
        return result["messages"][-1].content

    async def get_history(self, session_id: str) -> list:
        """获取会话历史"""
        # 确保已初始化
        await self._ensure_initialized()

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
        # 确保已初始化
        await self._ensure_initialized()

        config = {"configurable": {"thread_id": session_id}}
        await self.graph.aupdate_state(config, {"messages": []})

    async def list_all_sessions(self):
        """列出所有session_id（仅SQLite支持）"""
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()
            try:
                await cursor.execute("""
                    SELECT DISTINCT thread_id
                    FROM checkpoints
                    WHERE thread_id IS NOT NULL AND thread_id != ''
                """)
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
            except Exception as e:
                print(f"Error in list_all_sessions: {e}")
                return []

    async def get_database_stats(self):
        """获取数据库统计信息"""
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()
            try:
                # 获取总checkpoint数
                await cursor.execute("SELECT COUNT(*) FROM checkpoints")
                row = await cursor.fetchone()
                total_checkpoints = row[0] if row else 0

                # 获取唯一session数
                await cursor.execute("""
                    SELECT COUNT(DISTINCT thread_id)
                    FROM checkpoints
                    WHERE thread_id IS NOT NULL AND thread_id != ''
                """)
                row = await cursor.fetchone()
                total_sessions = row[0] if row else 0

                # 获取数据库文件大小
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

                # 获取每个session的详细信息
                await cursor.execute("""
                    SELECT
                        thread_id,
                        COUNT(*) as checkpoint_count
                    FROM checkpoints
                    WHERE thread_id IS NOT NULL AND thread_id != ''
                    GROUP BY thread_id
                    ORDER BY checkpoint_count DESC
                """)
                rows = await cursor.fetchall()

                sessions_detail = []
                for row in rows:
                    sessions_detail.append({
                        "session_id": row[0],
                        "checkpoint_count": row[1]
                    })

                return {
                    "total_sessions": total_sessions,
                    "total_checkpoints": total_checkpoints,
                    "database_size_bytes": db_size,
                    "database_size_mb": round(db_size / 1024 / 1024, 2),
                    "database_path": os.path.abspath(self.db_path),
                    "sessions": sessions_detail
                }

            except Exception as e:
                return {
                    "error": str(e),
                    "total_sessions": 0,
                    "total_checkpoints": 0,
                    "database_size_bytes": 0,
                    "database_size_mb": 0,
                    "database_path": os.path.abspath(self.db_path),
                    "sessions": []
                }

    async def get_session_detail(self, session_id: str):
        """获取指定session的详细信息"""
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()
            try:
                # 获取session的checkpoint信息
                await cursor.execute("""
                    SELECT
                        checkpoint_id,
                        parent_checkpoint_id,
                        checkpoint
                    FROM checkpoints
                    WHERE thread_id = ?
                    ORDER BY checkpoint_id DESC
                """, (session_id,))

                rows = await cursor.fetchall()
                checkpoints = []
                for row in rows:
                    checkpoints.append({
                        "checkpoint_id": row[0],
                        "parent_checkpoint_id": row[1],
                        "has_data": row[2] is not None
                    })

                return {
                    "session_id": session_id,
                    "checkpoint_count": len(checkpoints),
                    "checkpoints": checkpoints
                }

            except Exception as e:
                return {
                    "session_id": session_id,
                    "error": str(e),
                    "checkpoint_count": 0,
                    "checkpoints": []
                }
