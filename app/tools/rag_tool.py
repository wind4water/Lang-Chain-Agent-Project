"""
RAG 知识库搜索工具

作为 LangChain Tool 集成到 Agent 中，让 Agent 可以自动调用知识库
"""
from langchain.tools import BaseTool
from typing import Optional, Type, Any
from pydantic import BaseModel, Field, ConfigDict
import logging

logger = logging.getLogger(__name__)


class RAGSearchInput(BaseModel):
    """RAG 搜索工具的输入参数"""
    query: str = Field(description="要在知识库中搜索的问题或关键词")


class RAGSearchTool(BaseTool):
    """RAG 知识库搜索工具"""

    name: str = "knowledge_base_search"
    description: str = """在公司知识库中搜索信息。适用于以下场景：
    - 查询产品文档、使用说明
    - 查找技术文档、API 文档
    - 搜索公司内部规范、流程
    - 了解系统配置、功能特性
    - 了解某一个员工的基本信息

    输入：具体的问题或关键词
    输出：基于知识库的答案及来源

    示例：
    - "系统支持哪些文档格式？"
    - "如何配置 RAG 功能？"
    - "API 的认证方式是什么？"
    """
    args_schema: Type[BaseModel] = RAGSearchInput

    # Pydantic v2 配置：允许任意类型
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # RAG 系统实例（运行时注入）
    rag_system: Any = None

    def __init__(self):
        super().__init__()
        # 延迟导入，避免循环依赖
        from app.rag import rag_system
        self.rag_system = rag_system

    def _run(self, query: str) -> str:
        """同步执行（LangChain Tool 要求实现，但不推荐使用）"""
        import asyncio
        try:
            # 同步环境下运行异步代码
            return asyncio.run(self._arun(query))
        except Exception as e:
            logger.error(f"RAG 工具同步调用失败: {e}")
            return f"知识库搜索失败: {str(e)}"

    async def _arun(self, query: str) -> str:
        """异步执行 RAG 搜索"""
        try:
            # 检查 RAG 系统是否初始化
            if not self.rag_system._initialized:
                logger.warning("RAG 系统未初始化")
                return "❌ 知识库系统未启用。请联系管理员配置 RAG_ENABLED=true"

            logger.info(f"🔍 RAG 工具被调用: {query}")

            # 调用 RAG 系统查询（带来源）
            result = await self.rag_system.query(
                question=query,
                with_sources=True
            )

            # 构建返回结果
            answer = result.get('answer', '未找到相关信息')
            sources = result.get('sources', [])
            source_count = result.get('source_count', 0)

            # 格式化输出
            response = f"📚 知识库回答：\n{answer}"

            # 附加来源信息
            if source_count > 0 and sources:
                response += f"\n\n📄 参考来源（共 {source_count} 个）："
                for i, source in enumerate(sources[:3], 1):  # 最多显示3个来源
                    filename = source.get('filename', '未知')
                    preview = source.get('content_preview', '')[:100]
                    response += f"\n{i}. {filename}"
                    if preview:
                        response += f"\n   预览: {preview}..."

                if source_count > 3:
                    response += f"\n   ... 还有 {source_count - 3} 个来源"

            logger.info(f"✅ RAG 工具返回成功，来源数: {source_count}")
            return response

        except Exception as e:
            error_msg = f"知识库搜索出错: {str(e)}"
            logger.error(f"❌ RAG 工具执行失败: {e}")
            return f"❌ {error_msg}"
