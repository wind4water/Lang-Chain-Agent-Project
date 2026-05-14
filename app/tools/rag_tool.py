"""
RAG 知识库搜索工具

作为 LangChain Tool 集成到 Agent 中，让 Agent 可以自动调用知识库
"""
from langchain.tools import BaseTool
from typing import Optional, Type, Any, Dict
from pydantic import BaseModel, Field, ConfigDict
import logging
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class RAGSearchInput(BaseModel):
    """RAG 搜索工具的输入参数"""
    query: str = Field(description="要在知识库中搜索的问题或关键词")
    original_query: Optional[str] = Field(
        default=None,
        description="用户原始问题全文。调用方应透传原句，用于查询漂移保护。"
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="可选元数据过滤条件（Chroma filter），例如 {'filename': 'README.md'}"
    )


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

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"[\s\W_]+", "", (text or "").lower(), flags=re.UNICODE)

    def _is_suspicious_rewrite(self, query: str, original_query: str) -> bool:
        normalized_query = self._normalize_text(query)
        normalized_original = self._normalize_text(original_query)
        if not normalized_query or not normalized_original:
            return False

        # 明显包含关系一般认为安全（例如原句基础上的轻微补充）
        if normalized_query in normalized_original or normalized_original in normalized_query:
            return False

        similarity = SequenceMatcher(None, normalized_query, normalized_original).ratio()
        length_drop = (len(normalized_original) - len(normalized_query)) / max(len(normalized_original), 1)
        extra_chars = [ch for ch in normalized_query if ch not in normalized_original]
        extra_ratio = len(extra_chars) / max(len(normalized_query), 1)

        keyword_drift = (
            ("口头禅" in original_query and "口头禅" not in query) or
            ("回复" in original_query and "回复" not in query)
        )

        return keyword_drift or (similarity < 0.60 and (length_drop > 0.25 or extra_ratio > 0.35))

    def _pick_effective_query(self, query: str, original_query: Optional[str]) -> str:
        if not original_query:
            return query

        if len((query or "").strip()) < 3:
            logger.warning("⚠️ 检测到过短 query，回退 original_query: %s", original_query)
            return original_query

        if self._is_suspicious_rewrite(query, original_query):
            logger.warning(
                "⚠️ 检测到 query 漂移，回退 original_query。query=%s, original_query=%s",
                query,
                original_query,
            )
            return original_query

        return query

    def _run(
        self,
        query: str,
        original_query: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> str:
        """同步执行（LangChain Tool 要求实现，但不推荐使用）"""
        import asyncio
        try:
            # 同步环境下运行异步代码
            return asyncio.run(self._arun(query, original_query, metadata_filter))
        except Exception as e:
            logger.error(f"RAG 工具同步调用失败: {e}")
            return f"知识库搜索失败: {str(e)}"

    async def _arun(
        self,
        query: str,
        original_query: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> str:
        """异步执行 RAG 搜索"""
        try:
            # 检查 RAG 系统是否初始化
            if not self.rag_system._initialized:
                logger.warning("RAG 系统未初始化")
                return "❌ 知识库系统未启用。请联系管理员配置 RAG_ENABLED=true"

            effective_query = self._pick_effective_query(query=query, original_query=original_query)
            logger.info(
                "🔧 ToolStart: name=%s, query=%s, effective_query=%s, original_query=%s, metadata_filter=%s",
                self.name,
                query,
                effective_query,
                original_query,
                metadata_filter,
            )

            # 调用 RAG 系统查询（带来源）
            result = await self.rag_system.query(
                question=effective_query,
                with_sources=True,
                metadata_filter=metadata_filter
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

            logger.info(
                "🔧 ToolEnd: name=%s, source_count=%s, answer_preview=%s, output_preview=%s",
                self.name,
                source_count,
                (answer[:120] + "...") if len(answer) > 120 else answer,
                (response[:200] + "...") if len(response) > 200 else response,
            )
            return response

        except Exception as e:
            error_msg = f"知识库搜索出错: {str(e)}"
            logger.error("🔧 ToolError: name=%s, error=%s", self.name, e)
            return f"❌ {error_msg}"
