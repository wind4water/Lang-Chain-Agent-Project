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
from app.token_usage import record_llm_token_usage

# ============ Langfuse 集成 ============
from typing import Optional
import logging

# 条件导入 Langfuse（不强制依赖）
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None

logger = logging.getLogger(__name__)

# 抑制 Langfuse 内部错误的日志输出
logging.getLogger("langfuse").setLevel(logging.ERROR)
# ========================================


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

        # ============ Langfuse 配置和初始化 ============
        self.langfuse_enabled = False
        self.langfuse_client = None
        self.langfuse_public_key = None
        self.langfuse_secret_key = None
        self.langfuse_host = None

        # 检查是否启用 Langfuse
        if os.getenv("LANGFUSE_ENABLED", "false").lower() == "true":
            if not LANGFUSE_AVAILABLE:
                logger.warning("⚠️ Langfuse 已启用但未安装。请运行: pip install langfuse")
                print("⚠️ Langfuse 未安装，监控功能已禁用")
            else:
                try:
                    # 验证必需配置
                    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
                    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

                    if not public_key or not secret_key:
                        logger.warning("⚠️ Langfuse 配置不完整（缺少 PUBLIC_KEY 或 SECRET_KEY）")
                        print("⚠️ Langfuse 配置不完整，监控功能已禁用")
                    elif public_key.startswith("pk-lf-your-") or secret_key.startswith("sk-lf-your-"):
                        logger.warning("⚠️ Langfuse 配置为示例值，请更新为真实密钥")
                        print("⚠️ Langfuse 配置为示例值，监控功能已禁用")
                    else:
                        # 保存配置
                        self.langfuse_public_key = public_key
                        self.langfuse_secret_key = secret_key
                        self.langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

                        # 初始化 Langfuse 客户端 (v2.x API)
                        self.langfuse_client = Langfuse(
                            public_key=public_key,
                            secret_key=secret_key,
                            host=self.langfuse_host,
                        )
                        self.langfuse_enabled = True
                        self.langfuse_sample_rate = float(os.getenv("LANGFUSE_SAMPLE_RATE", "1.0"))

                        logger.info("✅ Langfuse 监控已启用")
                        print(f"✅ Langfuse 监控已启用 (采样率: {self.langfuse_sample_rate*100:.0f}%)")

                except Exception as e:
                    logger.error(f"⚠️ Langfuse 初始化失败: {e}")
                    print(f"⚠️ Langfuse 初始化失败: {e}")
                    self.langfuse_enabled = False
                    self.langfuse_client = None
        else:
            logger.info("ℹ️ Langfuse 监控未启用")
        # =============================================

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

    def _create_langfuse_callback(self, session_id: str, user_message: str = None):
        """返回 Langfuse callback handler

        Langfuse 2.x 使用 OpenTelemetry 集成，不再使用 CallbackHandler。
        这里返回 None，追踪通过手动方式在 chat 方法中处理。

        Args:
            session_id: 会话ID
            user_message: 用户消息

        Returns:
            None (Langfuse 2.x 不使用 callback handler)
        """
        if not self.langfuse_enabled or not self.langfuse_client:
            return None

        # 采样控制
        import random
        sample = random.random()
        if sample > self.langfuse_sample_rate:
            return None

        # 记录追踪信息
        trace_name = f"chat_{session_id}"
        if user_message:
            preview = user_message[:50] + "..." if len(user_message) > 50 else user_message
            trace_name = preview

        logger.info("🔵 Langfuse 追踪: session_id=%s, trace_name=%s",
                   session_id, trace_name)

        # Langfuse 2.x 不使用 callback handler，返回 None
        # 追踪在 chat 方法中通过 langfuse_client 手动处理
        return None

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

        # 创建 token_usage 表（新增）
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_id TEXT,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                model_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_session
            ON token_usage(session_id)
        """)

        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_date
            ON token_usage(created_at)
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
        """处理用户消息（包含Token统计和Langfuse追踪）"""
        await self._ensure_initialized()

        # ============ Langfuse 2.x: 手动追踪 ============
        observation = None
        if self.langfuse_enabled and self.langfuse_client:
            import random
            if random.random() <= self.langfuse_sample_rate:
                try:
                    trace_name = message[:50] + "..." if len(message) > 50 else message
                    observation = self.langfuse_client.start_observation(
                        name=trace_name,
                        as_type="generation",
                        model=self.llm.model_name,
                        input={"message": message},
                        metadata={
                            "session_id": session_id,
                            "enable_tools": self.enable_tools,
                        },
                    )
                    logger.info("🟢 Langfuse generation 已创建")
                except Exception as e:
                    logger.warning(f"Langfuse generation 创建失败: {e}")
        # ===============================================

        config = {
            "configurable": {"thread_id": session_id},
        }
        user_message = HumanMessage(content=message)

        # 记录请求开始时间
        import time
        start_time = time.time()

        logger.info("📤 开始调用 LangGraph: session_id=%s", session_id)

        result = await self.graph.ainvoke(
            {"messages": [user_message]},
            config=config
        )

        elapsed = time.time() - start_time
        logger.info("📥 LangGraph 调用完成: elapsed=%.2fs", elapsed)

        # 尝试从结果中提取Token使用信息（累计多次模型调用）
        prompt_tokens = 0
        completion_tokens = 0
        usage_found = False

        # 查找最后一条AI消息，尝试获取usage信息
        ai_message = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                ai_message = msg
                break

        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and hasattr(msg, 'response_metadata') and msg.response_metadata:
                usage = msg.response_metadata.get('token_usage') or msg.response_metadata.get('usage')
                if usage:
                    usage_found = True
                    prompt_tokens += usage.get('prompt_tokens', 0)
                    completion_tokens += usage.get('completion_tokens', 0)

        # 如果无法从metadata中获取，则进行简单估算
        if not usage_found and prompt_tokens == 0 and completion_tokens == 0:
            prompt_tokens = len(message) // 3
            if ai_message:
                completion_tokens = len(ai_message.content) // 3

        if prompt_tokens > 0 or completion_tokens > 0:
            record_llm_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model_name=self.llm.model_name
            )

        # 保存Token使用记录
        if prompt_tokens > 0 or completion_tokens > 0:
            try:
                await self._save_token_usage(
                    session_id=session_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    model_name=self.llm.model_name
                )
            except Exception as e:
                print(f"⚠️ Token统计保存失败: {e}")

        # ============ Langfuse 2.x: 结束追踪 ============
        if observation:
            try:
                response_text = ai_message.content if ai_message else ""
                observation.update(
                    output={"response": response_text},
                    usage_details={
                        "input": prompt_tokens,
                        "output": completion_tokens,
                    } if prompt_tokens > 0 or completion_tokens > 0 else None,
                )
                observation.end()
                logger.info("✅ Langfuse generation 已完成")
            except Exception as e:
                logger.warning(f"Langfuse generation 结束失败: {e}")
        # ============================================

        # 返回最后一条 AI 消息
        if ai_message:
            return ai_message.content

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

    async def _save_token_usage(self, session_id: str, prompt_tokens: int,
                                 completion_tokens: int, model_name: str,
                                 message_id: str = None):
        """保存Token使用记录

        Args:
            session_id: 会话ID
            prompt_tokens: 提示词token数
            completion_tokens: 完成token数
            model_name: 模型名称
            message_id: 消息ID（可选）
        """
        total_tokens = prompt_tokens + completion_tokens

        # 根据模型计算成本（单位：美元/1000 tokens）
        cost_per_1k = {
            # OpenAI 模型
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'gpt-4o': {'input': 0.005, 'output': 0.015},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
            # 智谱AI 模型
            'glm-4-flash': {'input': 0.0001, 'output': 0.0001},
            'glm-4': {'input': 0.001, 'output': 0.001},
            # 默认价格（未知模型）
            'default': {'input': 0, 'output': 0}
        }

        pricing = cost_per_1k.get(model_name, cost_per_1k['default'])
        cost = (prompt_tokens / 1000 * pricing['input'] +
                completion_tokens / 1000 * pricing['output'])

        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                INSERT INTO token_usage
                (session_id, message_id, prompt_tokens, completion_tokens, total_tokens, cost_usd, model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (session_id, message_id, prompt_tokens, completion_tokens, total_tokens, cost, model_name))
            await conn.commit()

    async def get_token_stats(self, session_id: str):
        """获取指定会话的Token使用统计

        Args:
            session_id: 会话ID

        Returns:
            统计信息字典
        """
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()
            try:
                # 获取总计
                await cursor.execute("""
                    SELECT
                        COUNT(*) as request_count,
                        SUM(prompt_tokens) as total_prompt_tokens,
                        SUM(completion_tokens) as total_completion_tokens,
                        SUM(total_tokens) as total_tokens,
                        SUM(cost_usd) as total_cost,
                        model_name
                    FROM token_usage
                    WHERE session_id = ?
                    GROUP BY model_name
                """, (session_id,))

                rows = await cursor.fetchall()

                if not rows:
                    return {
                        "session_id": session_id,
                        "request_count": 0,
                        "total_prompt_tokens": 0,
                        "total_completion_tokens": 0,
                        "total_tokens": 0,
                        "total_cost_usd": 0.0,
                        "by_model": []
                    }

                # 汇总所有模型的数据
                total_requests = sum(row[0] for row in rows)
                total_prompt = sum(row[1] or 0 for row in rows)
                total_completion = sum(row[2] or 0 for row in rows)
                total = sum(row[3] or 0 for row in rows)
                total_cost = sum(row[4] or 0 for row in rows)

                by_model = [
                    {
                        "model_name": row[5],
                        "request_count": row[0],
                        "prompt_tokens": row[1] or 0,
                        "completion_tokens": row[2] or 0,
                        "total_tokens": row[3] or 0,
                        "cost_usd": round(row[4] or 0, 6)
                    }
                    for row in rows
                ]

                return {
                    "session_id": session_id,
                    "request_count": total_requests,
                    "total_prompt_tokens": total_prompt,
                    "total_completion_tokens": total_completion,
                    "total_tokens": total,
                    "total_cost_usd": round(total_cost, 6),
                    "by_model": by_model
                }

            except Exception as e:
                return {
                    "session_id": session_id,
                    "error": str(e),
                    "request_count": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0
                }

    async def get_daily_token_stats(self, date: str = None):
        """获取每日Token使用汇总

        Args:
            date: 日期字符串 (YYYY-MM-DD)，默认为今天

        Returns:
            当日统计信息
        """
        from datetime import datetime, timedelta

        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()
            try:
                await cursor.execute("""
                    SELECT
                        COUNT(DISTINCT session_id) as unique_sessions,
                        COUNT(*) as total_requests,
                        SUM(prompt_tokens) as total_prompt_tokens,
                        SUM(completion_tokens) as total_completion_tokens,
                        SUM(total_tokens) as total_tokens,
                        SUM(cost_usd) as total_cost
                    FROM token_usage
                    WHERE DATE(created_at) = ?
                """, (date,))

                row = await cursor.fetchone()

                if not row or row[0] is None:
                    return {
                        "date": date,
                        "unique_sessions": 0,
                        "total_requests": 0,
                        "total_tokens": 0,
                        "total_cost_usd": 0.0
                    }

                return {
                    "date": date,
                    "unique_sessions": row[0] or 0,
                    "total_requests": row[1] or 0,
                    "total_prompt_tokens": row[2] or 0,
                    "total_completion_tokens": row[3] or 0,
                    "total_tokens": row[4] or 0,
                    "total_cost_usd": round(row[5] or 0, 6)
                }

            except Exception as e:
                return {
                    "date": date,
                    "error": str(e),
                    "unique_sessions": 0,
                    "total_requests": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0
                }

    async def get_monthly_token_stats(self, year_month: str = None):
        """获取每月Token使用汇总

        Args:
            year_month: 年月字符串 (YYYY-MM)，默认为当月

        Returns:
            当月统计信息
        """
        from datetime import datetime

        if year_month is None:
            year_month = datetime.now().strftime("%Y-%m")

        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()
            try:
                await cursor.execute("""
                    SELECT
                        COUNT(DISTINCT session_id) as unique_sessions,
                        COUNT(*) as total_requests,
                        SUM(prompt_tokens) as total_prompt_tokens,
                        SUM(completion_tokens) as total_completion_tokens,
                        SUM(total_tokens) as total_tokens,
                        SUM(cost_usd) as total_cost
                    FROM token_usage
                    WHERE strftime('%Y-%m', created_at) = ?
                """, (year_month,))

                row = await cursor.fetchone()

                if not row or row[0] is None:
                    return {
                        "year_month": year_month,
                        "unique_sessions": 0,
                        "total_requests": 0,
                        "total_tokens": 0,
                        "total_cost_usd": 0.0
                    }

                return {
                    "year_month": year_month,
                    "unique_sessions": row[0] or 0,
                    "total_requests": row[1] or 0,
                    "total_prompt_tokens": row[2] or 0,
                    "total_completion_tokens": row[3] or 0,
                    "total_tokens": row[4] or 0,
                    "total_cost_usd": round(row[5] or 0, 6)
                }

            except Exception as e:
                return {
                    "year_month": year_month,
                    "error": str(e),
                    "unique_sessions": 0,
                    "total_requests": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0
                }

    async def chat_stream(self, message: str, session_id: str):
        """流式输出对话内容（支持Langfuse追踪）

        Args:
            message: 用户消息
            session_id: 会话ID

        Yields:
            字符串片段（逐字输出）
        """
        await self._ensure_initialized()

        # ============ Langfuse 2.x: 手动追踪 ============
        observation = None
        if self.langfuse_enabled and self.langfuse_client:
            import random
            if random.random() <= self.langfuse_sample_rate:
                try:
                    trace_name = message[:50] + "..." if len(message) > 50 else message
                    observation = self.langfuse_client.start_observation(
                        name=trace_name,
                        as_type="generation",
                        model=self.llm.model_name,
                        input={"message": message},
                        metadata={"session_id": session_id, "stream": True},
                    )
                    logger.info("🟢 Langfuse generation 已创建（流式）")
                except Exception as e:
                    logger.warning(f"Langfuse generation 创建失败: {e}")
        # ===============================================

        config = {
            "configurable": {"thread_id": session_id},
        }
        user_message = HumanMessage(content=message)

        # 用于累积完整响应（用于后续保存Token统计）
        full_response = ""
        prompt_tokens = 0
        completion_tokens = 0

        logger.info("📤 开始流式调用 LangGraph: session_id=%s", session_id)

        try:
            # 使用 astream_events 获取流式事件
            async for event in self.graph.astream_events(
                {"messages": [user_message]},
                config=config,
                version="v2"  # 使用v2版本的事件API
            ):
                kind = event["event"]

                # 处理不同类型的事件
                if kind == "on_chat_model_stream":
                    # LLM流式输出的内容片段
                    content = event["data"]["chunk"].content
                    if content:
                        full_response += content
                        yield content

                elif kind == "on_chat_model_end":
                    # LLM完成，尝试获取usage信息
                    if "response_metadata" in event["data"]["output"]:
                        metadata = event["data"]["output"]["response_metadata"]
                        usage = metadata.get('token_usage') or metadata.get('usage')
                        if usage:
                            prompt_tokens += usage.get('prompt_tokens', 0)
                            completion_tokens += usage.get('completion_tokens', 0)

            # 流式输出完成后，保存Token统计
            if prompt_tokens == 0 and completion_tokens == 0:
                prompt_tokens = len(message) // 3
                completion_tokens = len(full_response) // 3

            if prompt_tokens > 0 or completion_tokens > 0:
                record_llm_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    model_name=self.llm.model_name
                )

            if prompt_tokens > 0 or completion_tokens > 0:
                try:
                    await self._save_token_usage(
                        session_id=session_id,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        model_name=self.llm.model_name
                    )
                except Exception as e:
                    print(f"⚠️ Token统计保存失败: {e}")

            # ============ Langfuse 2.x: 结束追踪 ============
            if observation:
                try:
                    observation.update(
                        output={"response": full_response},
                        usage_details={
                            "input": prompt_tokens,
                            "output": completion_tokens,
                        } if prompt_tokens > 0 or completion_tokens > 0 else None,
                    )
                    observation.end()
                    logger.info("✅ Langfuse generation 已完成（流式）")
                except Exception as e:
                    logger.warning(f"Langfuse generation 结束失败: {e}")
            # ============================================

        except Exception as e:
            # 流式输出过程中的错误
            yield f"\n\n[错误: {str(e)}]"

            # ============ Langfuse: 错误时更新 observation ============
            if observation:
                try:
                    observation.update(output={"error": str(e)})
                    observation.end()
                except Exception:
                    pass
            # ===============================================


