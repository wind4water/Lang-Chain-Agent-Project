"""
工具加载器
负责加载和管理所有可用的工具
"""

from typing import List, Optional
from langchain_core.tools import BaseTool
import logging
import os

from .builtin import BUILTIN_TOOLS

logger = logging.getLogger(__name__)


class ToolLoader:
    """工具加载器类

    管理工具的加载、注册和获取
    """

    def __init__(self):
        self._tools: List[BaseTool] = []
        self._tool_names: set = set()

    def register_tool(self, tool: BaseTool) -> bool:
        """注册单个工具

        Args:
            tool: 要注册的工具

        Returns:
            是否注册成功
        """
        if tool.name in self._tool_names:
            logger.warning(f"工具 {tool.name} 已存在，跳过注册")
            return False

        self._tools.append(tool)
        self._tool_names.add(tool.name)
        logger.info(f"✅ 注册工具: {tool.name}")
        return True

    def register_tools(self, tools: List[BaseTool]) -> int:
        """批量注册工具

        Args:
            tools: 工具列表

        Returns:
            成功注册的工具数量
        """
        count = 0
        for tool in tools:
            if self.register_tool(tool):
                count += 1
        return count

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """根据名称获取工具

        Args:
            name: 工具名称

        Returns:
            工具对象，如果不存在返回 None
        """
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None

    def get_all_tools(self) -> List[BaseTool]:
        """获取所有已注册的工具

        Returns:
            工具列表
        """
        return self._tools.copy()

    def list_tool_names(self) -> List[str]:
        """列出所有工具名称

        Returns:
            工具名称列表
        """
        return [tool.name for tool in self._tools]

    def clear(self):
        """清空所有工具"""
        self._tools.clear()
        self._tool_names.clear()
        logger.info("已清空所有工具")


# 全局工具加载器实例
_global_loader = ToolLoader()


def load_builtin_tools() -> List[BaseTool]:
    """加载所有内置工具

    Returns:
        内置工具列表
    """
    logger.info(f"加载 {len(BUILTIN_TOOLS)} 个内置工具...")
    return BUILTIN_TOOLS


def load_all_tools(include_builtin: bool = True) -> List[BaseTool]:
    """加载所有可用的工具

    Args:
        include_builtin: 是否包含内置工具，默认 True

    Returns:
        所有工具的列表
    """
    tools = []

    # 1. 加载内置工具
    if include_builtin:
        builtin = load_builtin_tools()
        tools.extend(builtin)
        logger.info(f"✅ 加载了 {len(builtin)} 个内置工具")

    # 2. 加载 RAG 知识库搜索工具（如果启用）
    if os.getenv("RAG_ENABLED", "false").lower() == "true":
        try:
            from .rag_tool import RAGSearchTool
            rag_tool = RAGSearchTool()
            tools.append(rag_tool)
            logger.info(f"✅ 加载了 RAG 知识库搜索工具")
        except Exception as e:
            logger.warning(f"⚠️  加载 RAG 工具失败: {e}")
            logger.warning("   RAG 工具将不可用，但不影响其他功能")

    # 3. 未来可以在这里加载自定义工具
    # custom_tools = load_custom_tools()
    # tools.extend(custom_tools)

    # 4. 未来可以在这里加载 MCP 工具
    # mcp_tools = load_mcp_tools()
    # tools.extend(mcp_tools)

    logger.info(f"📦 总共加载了 {len(tools)} 个工具")
    return tools


def get_tool_descriptions() -> str:
    """获取所有工具的描述信息

    Returns:
        格式化的工具描述字符串
    """
    tools = load_all_tools()
    descriptions = []

    descriptions.append(f"共有 {len(tools)} 个可用工具：\n")

    for i, tool in enumerate(tools, 1):
        descriptions.append(f"{i}. **{tool.name}**")
        descriptions.append(f"   描述: {tool.description}")
        descriptions.append("")

    return "\n".join(descriptions)
