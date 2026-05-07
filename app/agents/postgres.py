"""
PostgreSQL持久化版本 - 适合生产环境
支持分布式部署、高并发、多实例共享
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import os


class State(TypedDict):
    """对话状态定义"""
    messages: Annotated[list, add_messages]


class PostgresAgent:
    """使用PostgreSQL持久化的对话Agent"""

    def __init__(self, pg_url: str = None):
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

        # PostgreSQL连接URL
        self.pg_url = pg_url or os.getenv(
            "POSTGRES_URL",
            "postgresql://langchain:langchain@localhost:5432/langchain_db"
        )

        # checkpointer将在首次使用时异步初始化
        self.checkpointer = None
        self.graph = None

        print(f"✅ 配置PostgreSQL存储: {self.pg_url.split('@')[1] if '@' in self.pg_url else self.pg_url}")

    async def _ensure_initialized(self):
        """确保checkpointer已初始化"""
        if self.graph is None:
            # 创建异步PostgreSQL checkpointer
            self.checkpointer = AsyncPostgresSaver.from_conn_string(self.pg_url)

            # 初始化数据库表（如果不存在）
            await self.checkpointer.setup()

            # 构建StateGraph
            workflow = StateGraph(State)
            workflow.add_node("chat", self._chat_node)
            workflow.add_edge(START, "chat")
            workflow.add_edge("chat", END)

            self.graph = workflow.compile(checkpointer=self.checkpointer)
            print("✅ PostgreSQL checkpointer初始化完成")

    def _chat_node(self, state: State):
        """对话节点"""
        messages = state["messages"]
        response = self.chain.invoke({"messages": messages})
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

        return result["messages"][-1].content

    async def get_history(self, session_id: str) -> list:
        """获取会话历史"""
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
        await self._ensure_initialized()

        config = {"configurable": {"thread_id": session_id}}
        await self.graph.aupdate_state(config, {"messages": []})
