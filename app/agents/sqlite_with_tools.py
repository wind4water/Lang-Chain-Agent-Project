"""
SqliteAgent with Tool Calling
支持工具调用的 SQLite 持久化 Agent

核心特性：
1. Tool Calling - 自动调用工具完成任务
2. SQLite 持久化 - 会话数据保存到数据库
3. 上下文压缩 - 支持多种压缩策略
"""

from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
import os
import aiosqlite

from app.tools.loader import load_all_tools


class State(TypedDict):
    """对话状态定义"""
    messages: Annotated[list, add_messages]


class SqliteAgentWithTools:
    """支持工具调用的持久化对话Agent"""

    def __init__(self, db_path: str = "checkpoints/conversations.db", enable_tools: bool = True):
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

        # 保存数据库路径
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # 工具系统
        self.enable_tools = enable_tools
        self.tools: List[BaseTool] = []
        if enable_tools:
            self.tools = load_all_tools()
            print(f"🔧 加载了 {len(self.tools)} 个工具:")
            for tool in self.tools:
                print(f"   - {tool.name}: {tool.description}")

        # 读取压缩策略配置
        self.compression_strategy = os.getenv("CONTEXT_COMPRESSION_STRATEGY", "none")
        self.window_size = int(os.getenv("COMPRESSION_WINDOW_SIZE", "10"))
        self.max_tokens = int(os.getenv("COMPRESSION_MAX_TOKENS", "4000"))
        self.summary_threshold = int(os.getenv("COMPRESSION_SUMMARY_THRESHOLD", "20"))

        # 如果使用摘要策略，初始化摘要LLM
        self.summary_llm = None
        if self.compression_strategy == "summary":
            self.summary_llm = ChatOpenAI(
                model=os.getenv("SUMMARY_MODEL_NAME", "gpt-4o-mini"),
                temperature=0.3,
                api_key=api_key,
                base_url=os.getenv("OPENAI_BASE_URL")
            )

        # checkpointer 和 graph 将在首次使用时初始化
        self._conn = None
        self.checkpointer = None
        self.graph = None

        print(f"✅ 配置持久化存储：{os.path.abspath(db_path)}")
        print(f"✅ 上下文压缩策略: {self.compression_strategy}")
        print(f"✅ 工具调用: {'启用' if enable_tools else '禁用'}")

    async def _ensure_initialized(self):
        """确保checkpointer和graph已初始化"""
        if self.graph is None:
            # 创建数据库连接
            self._conn = await aiosqlite.connect(self.db_path)
            self.checkpointer = AsyncSqliteSaver(self._conn)
            await self._init_tables()

            # 使用 create_react_agent 创建支持工具的 agent
            if self.enable_tools and len(self.tools) > 0:
                # 绑定工具到 LLM
                llm_with_tools = self.llm.bind_tools(self.tools)

                # 创建系统消息
                system_message = """你是一个智能助手，可以使用工具来帮助回答问题。

当你需要获取实时信息或执行特定任务时，请使用可用的工具。

可用工具:
{tools}

使用工具的步骤:
1. 判断是否需要工具（用户的问题是否需要实时数据或特定功能）
2. 选择合适的工具
3. 调用工具并获取结果
4. 根据结果回答用户的问题

如果不需要工具，直接回答即可。
"""

                # 使用 LangGraph 的 create_react_agent
                self.graph = create_react_agent(
                    llm_with_tools,
                    self.tools,
                    checkpointer=self.checkpointer,
                    state_modifier=system_message,
                )

                print("✅ 使用 create_react_agent 初始化（支持工具调用）")
            else:
                # 不使用工具的普通 agent
                workflow = StateGraph(State)
                workflow.add_node("chat", self._chat_node)
                workflow.add_edge(START, "chat")
                workflow.add_edge("chat", END)

                self.graph = workflow.compile(checkpointer=self.checkpointer)
                print("✅ 使用 StateGraph 初始化（无工具）")

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

    def _compress_sliding_window(self, messages: list) -> list:
        """压缩策略1: 滑动窗口"""
        if len(messages) <= self.window_size * 2:
            return messages

        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        other_messages = [m for m in messages if not isinstance(m, SystemMessage)]
        recent_messages = other_messages[-(self.window_size * 2):]

        result = system_messages + recent_messages
        print(f"🔄 [滑动窗口] 压缩: {len(messages)} → {len(result)} 条消息")
        return result

    async def _apply_compression(self, messages: list) -> list:
        """应用压缩策略"""
        if self.compression_strategy == "none":
            return messages
        elif self.compression_strategy == "sliding_window":
            return self._compress_sliding_window(messages)
        else:
            return messages

    async def _chat_node(self, state: State):
        """普通对话节点（无工具时使用）"""
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.output_parsers import StrOutputParser

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="你是一个友好且乐于助人的AI助手。"),
            MessagesPlaceholder(variable_name="messages"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        messages = state["messages"]
        compressed_messages = await self._apply_compression(messages)

        response = await chain.ainvoke({"messages": compressed_messages})
        return {"messages": [AIMessage(content=response)]}

    async def chat(self, message: str, session_id: str) -> str:
        """处理用户消息"""
        await self._ensure_initialized()

        config = {"configurable": {"thread_id": session_id}}
        user_message = HumanMessage(content=message)

        result = await self.graph.ainvoke(
            {"messages": [user_message]},
            config=config
        )

        # 返回最后一条 AI 消息
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content

        return "没有收到有效回复"

    async def get_history(self, session_id: str) -> list:
        """获取会话历史"""
        await self._ensure_initialized()

        config = {"configurable": {"thread_id": session_id}}
        state = await self.graph.aget_state(config)

        if state and state.values.get("messages"):
            history = []
            for msg in state.values["messages"]:
                if isinstance(msg, HumanMessage):
                    history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    history.append({"role": "assistant", "content": msg.content})
                elif isinstance(msg, ToolMessage):
                    # 工具调用结果也记录到历史
                    history.append({
                        "role": "tool",
                        "tool_name": msg.name if hasattr(msg, "name") else "unknown",
                        "content": msg.content
                    })
            return history
        return []

    async def clear_history(self, session_id: str):
        """清除会话历史"""
        await self._ensure_initialized()
        config = {"configurable": {"thread_id": session_id}}
        await self.graph.aupdate_state(config, {"messages": []})

    async def list_all_sessions(self):
        """列出所有session_id"""
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
                await cursor.execute("SELECT COUNT(*) FROM checkpoints")
                row = await cursor.fetchone()
                total_checkpoints = row[0] if row else 0

                await cursor.execute("""
                    SELECT COUNT(DISTINCT thread_id)
                    FROM checkpoints
                    WHERE thread_id IS NOT NULL AND thread_id != ''
                """)
                row = await cursor.fetchone()
                total_sessions = row[0] if row else 0

                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

                await cursor.execute("""
                    SELECT thread_id, COUNT(*) as checkpoint_count
                    FROM checkpoints
                    WHERE thread_id IS NOT NULL AND thread_id != ''
                    GROUP BY thread_id
                    ORDER BY checkpoint_count DESC
                """)
                rows = await cursor.fetchall()

                sessions_detail = [
                    {"session_id": row[0], "checkpoint_count": row[1]}
                    for row in rows
                ]

                return {
                    "total_sessions": total_sessions,
                    "total_checkpoints": total_checkpoints,
                    "database_size_bytes": db_size,
                    "database_size_mb": round(db_size / 1024 / 1024, 2),
                    "database_path": os.path.abspath(self.db_path),
                    "tools_enabled": self.enable_tools,
                    "tools_count": len(self.tools),
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
                    "tools_enabled": self.enable_tools,
                    "tools_count": len(self.tools),
                    "sessions": []
                }

    async def get_session_detail(self, session_id: str):
        """获取指定session的详细信息"""
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()
            try:
                await cursor.execute("""
                    SELECT checkpoint_id, parent_checkpoint_id, checkpoint
                    FROM checkpoints
                    WHERE thread_id = ?
                    ORDER BY checkpoint_id DESC
                """, (session_id,))

                rows = await cursor.fetchall()
                checkpoints = [
                    {
                        "checkpoint_id": row[0],
                        "parent_checkpoint_id": row[1],
                        "has_data": row[2] is not None
                    }
                    for row in rows
                ]

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

    def list_available_tools(self) -> list:
        """列出所有可用的工具"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
            }
            for tool in self.tools
        ]
