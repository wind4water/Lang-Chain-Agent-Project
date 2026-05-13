"""
升级版Agent - 支持持久化存储到SQLite数据库
上下文会保存到磁盘文件，重启服务后仍然存在
支持3种上下文压缩策略：滑动窗口、Token计数、智能摘要
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
import httpx
import urllib3
import logging

logger = logging.getLogger(__name__)

# SSL 验证配置（默认启用）
_SSL_VERIFY = os.getenv("SSL_VERIFY", "true").lower() != "false"
_http_async_client = None
_http_client = None
if not _SSL_VERIFY:
    logger.warning("⚠️ SSL 证书验证已禁用（SSL_VERIFY=false）")
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _http_async_client = httpx.AsyncClient(verify=False)
    _http_client = httpx.Client(verify=False)


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
        llm_kwargs = {
            "model": os.getenv("MODEL_NAME", "gpt-4o-mini"),
            "temperature": 0.7,
            "api_key": api_key,
            "base_url": os.getenv("OPENAI_BASE_URL"),
        }
        if not _SSL_VERIFY:
            llm_kwargs["http_async_client"] = _http_async_client
            llm_kwargs["http_client"] = _http_client
        self.llm = ChatOpenAI(**llm_kwargs)

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

        # 读取压缩策略配置
        self.compression_strategy = os.getenv("CONTEXT_COMPRESSION_STRATEGY", "none")
        self.window_size = int(os.getenv("COMPRESSION_WINDOW_SIZE", "10"))
        self.max_tokens = int(os.getenv("COMPRESSION_MAX_TOKENS", "4000"))
        self.summary_threshold = int(os.getenv("COMPRESSION_SUMMARY_THRESHOLD", "20"))

        # 如果使用摘要策略，初始化摘要LLM（使用更便宜的模型）
        self.summary_llm = None
        if self.compression_strategy == "summary":
            summary_kwargs = {
                "model": os.getenv("SUMMARY_MODEL_NAME", "gpt-4o-mini"),
                "temperature": 0.3,
                "api_key": api_key,
                "base_url": os.getenv("OPENAI_BASE_URL"),
            }
            if not _SSL_VERIFY:
                summary_kwargs["http_async_client"] = _http_async_client
                summary_kwargs["http_client"] = _http_client
            self.summary_llm = ChatOpenAI(**summary_kwargs)

        # checkpointer 将在首次使用时异步初始化
        self._conn = None
        self.checkpointer = None
        self.graph = None

        print(f"✅ 配置持久化存储：{os.path.abspath(db_path)}")
        print(f"✅ 上下文压缩策略: {self.compression_strategy}")

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

    def _compress_sliding_window(self, messages: list) -> list:
        """压缩策略1: 滑动窗口 - 只保留最近N轮对话"""
        if len(messages) <= self.window_size * 2:
            return messages

        # 保留系统消息
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        other_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        # 保留最近N轮对话（N轮 = 2N条消息）
        recent_messages = other_messages[-(self.window_size * 2):]

        result = system_messages + recent_messages
        print(f"🔄 [滑动窗口] 压缩: {len(messages)} → {len(result)} 条消息")
        return result

    def _compress_token_limit(self, messages: list) -> list:
        """压缩策略2: Token计数 - 按token数量裁剪"""
        # 保留系统消息
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        other_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        # 从后往前累加，直到达到token限制
        trimmed = []
        token_count = 0

        for msg in reversed(other_messages):
            # 粗略估计：4个字符约等于1个token
            msg_tokens = len(str(msg.content)) // 4

            if token_count + msg_tokens > self.max_tokens:
                break

            trimmed.insert(0, msg)
            token_count += msg_tokens

        result = system_messages + trimmed
        print(f"🔄 [Token计数] 压缩: {len(messages)} → {len(result)} 条消息 (约{token_count} tokens)")
        return result

    async def _compress_summary(self, messages: list) -> list:
        """压缩策略3: 智能摘要 - 用LLM总结旧对话"""
        if len(messages) <= self.summary_threshold * 2:
            return messages

        # 保留系统消息
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        other_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        # 旧对话：需要摘要（除了最近3轮）
        old_messages = other_messages[:-(6)]
        recent_messages = other_messages[-(6):]

        if len(old_messages) > 0:
            # 生成摘要
            summary_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""请总结以下对话的关键信息，要求：
1. 提炼用户的主要问题和需求
2. 总结已经讨论的关键点
3. 记录重要的决策或结论
4. 用3-5句话概括，保留所有关键信息"""),
                MessagesPlaceholder(variable_name="messages")
            ])

            summary_chain = summary_prompt | self.summary_llm | StrOutputParser()

            try:
                summary_text = await summary_chain.ainvoke({"messages": old_messages})
                summary_msg = SystemMessage(content=f"📝 历史对话摘要：\n{summary_text}")

                result = system_messages + [summary_msg] + recent_messages
                print(f"🔄 [智能摘要] 压缩: {len(messages)} → {len(result)} 条消息 (摘要+最近3轮)")
                return result
            except Exception as e:
                print(f"⚠️ 摘要生成失败: {e}，回退到滑动窗口策略")
                return self._compress_sliding_window(messages)

        return messages

    async def _apply_compression(self, messages: list) -> list:
        """应用配置的压缩策略"""
        if self.compression_strategy == "none":
            return messages
        elif self.compression_strategy == "sliding_window":
            return self._compress_sliding_window(messages)
        elif self.compression_strategy == "token_limit":
            return self._compress_token_limit(messages)
        elif self.compression_strategy == "summary":
            return await self._compress_summary(messages)
        else:
            print(f"⚠️ 未知的压缩策略: {self.compression_strategy}，不进行压缩")
            return messages

    async def _chat_node(self, state: State):
        """对话节点"""
        messages = state["messages"]

        # 应用压缩策略
        compressed_messages = await self._apply_compression(messages)

        response = await self.chain.ainvoke({"messages": compressed_messages})
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
