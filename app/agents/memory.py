from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import os


class State(TypedDict):
    """对话状态定义"""
    messages: Annotated[list, add_messages]


class MemoryAgent:
    """LangChain对话Agent，使用LangGraph和Checkpoint"""

    def __init__(self):
        # 检查API密钥
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("sk-your-api-key") or api_key == "your-api-key-here":
            raise ValueError(
                "❌ 未配置API密钥！\n"
                "请按以下步骤操作：\n"
                "1. 编辑项目根目录的 .env 文件\n"
                "2. 将 OPENAI_API_KEY 设置为你的真实API密钥\n"
                "3. 保存后重新启动服务"
            )

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

        # 使用MemorySaver作为checkpoint（内存存储，重启后会丢失）
        # 如果需要持久化，可以用pickle定期保存到磁盘
        self.checkpointer = MemorySaver()

        # 构建StateGraph
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建对话图"""
        workflow = StateGraph(State)
        workflow.add_node("chat", self._chat_node)
        workflow.add_edge(START, "chat")
        workflow.add_edge("chat", END)
        return workflow.compile(checkpointer=self.checkpointer)

    def _chat_node(self, state: State):
        """对话节点 - 处理用户消息并生成回复"""
        messages = state["messages"]

        # 使用LLM处理消息
        response = self.chain.invoke({"messages": messages})

        # 返回新的状态
        return {"messages": [AIMessage(content=response)]}

    async def chat(self, message: str, session_id: str) -> str:
        """
        处理用户消息

        Args:
            message: 用户输入的消息
            session_id: 会话ID，用于checkpoint恢复对话历史

        Returns:
            AI的回复
        """
        # 配置，包含session_id用于checkpoint
        config = {"configurable": {"thread_id": session_id}}

        # 创建用户消息
        user_message = HumanMessage(content=message)

        # 调用图
        result = await self.graph.ainvoke(
            {"messages": [user_message]},
            config=config
        )

        # 返回最后一条消息（AI的回复）
        return result["messages"][-1].content

    async def get_history(self, session_id: str) -> list:
        """
        获取会话历史

        Args:
            session_id: 会话ID

        Returns:
            历史消息列表
        """
        config = {"configurable": {"thread_id": session_id}}

        # 获取checkpoint状态
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
