"""
RAG Chain - 检索增强生成链
"""
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import logging

from app.token_usage import record_llm_token_usage

logger = logging.getLogger(__name__)


class RAGChain:
    """RAG 问答链"""

    # 默认 Prompt 模板
    DEFAULT_PROMPT = """你是一个智能问答助手。请基于以下提供的上下文信息来回答用户的问题。

**重要规则**：
1. 只使用上下文中的信息回答问题
2. 如果上下文中没有相关信息，明确告诉用户"根据现有资料无法回答该问题"
3. 回答要准确、具体，引用上下文中的关键信息
4. 如果可能，说明信息的来源（如文件名）

上下文信息：
{context}

用户问题：{question}

请提供详细的回答："""

    def __init__(
        self,
        llm: ChatOpenAI,
        retriever,
        prompt_template: Optional[str] = None
    ):
        """
        初始化 RAG Chain

        Args:
            llm: 语言模型
            retriever: 检索器
            prompt_template: 自定义 Prompt 模板
        """
        self.llm = llm
        self.retriever = retriever
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT

        # 构建 Chain
        self.chain = self._build_chain()

        logger.info("✅ RAG Chain 初始化完成")

    def _record_usage(self, question: str, answer: str):
        """统一记录 token 使用（当前为近似统计）。"""
        record_llm_token_usage(
            prompt_tokens=len(question) // 3,
            completion_tokens=len(answer) // 3,
            model_name=getattr(self.llm, "model_name", "")
        )

    def _build_messages(self, question: str, docs: List[Document]) -> list:
        """基于问题与已检索文档构造提示消息。"""
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        context = self._format_docs(docs)
        return prompt.format_messages(context=context, question=question)

    def _invoke_from_docs(self, question: str, docs: List[Document]) -> str:
        """同步：复用已检索 docs 直接生成答案（不再二次检索）。"""
        messages = self._build_messages(question, docs)
        response = self.llm.invoke(messages)
        answer = StrOutputParser().invoke(response)
        self._record_usage(question, answer)
        return answer

    async def _ainvoke_from_docs(self, question: str, docs: List[Document]) -> str:
        """异步：复用已检索 docs 直接生成答案（不再二次检索）。"""
        messages = self._build_messages(question, docs)
        response = await self.llm.ainvoke(messages)
        answer = StrOutputParser().invoke(response)
        self._record_usage(question, answer)
        return answer

    def _build_chain(self):
        """构建 RAG Chain"""
        # 创建 Prompt
        prompt = ChatPromptTemplate.from_template(self.prompt_template)

        # 构建 Chain
        chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """
        格式化文档为上下文字符串

        Args:
            docs: 文档列表

        Returns:
            格式化的上下文字符串
        """
        if not docs:
            return "（未找到相关信息）"

        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            filename = doc.metadata.get("filename", "")

            # 格式化单个文档
            doc_text = f"【文档 {i}】"
            if filename:
                doc_text += f"（来源: {filename}）"
            doc_text += f"\n{doc.page_content}\n"

            formatted.append(doc_text)

        return "\n".join(formatted)

    def invoke(self, question: str) -> str:
        """
        同步调用 RAG Chain

        Args:
            question: 用户问题

        Returns:
            答案
        """
        logger.info(f"RAG 查询: {question}")

        try:
            answer = self.chain.invoke(question)
            # 当前实现通过 StrOutputParser 丢失了原始 usage，使用近似值统计请求级 token
            self._record_usage(question, answer)
            logger.info("✅ RAG 回答生成成功")
            return answer

        except Exception as e:
            logger.error(f"RAG 查询失败: {e}")
            raise

    async def ainvoke(self, question: str) -> str:
        """
        异步调用 RAG Chain

        Args:
            question: 用户问题

        Returns:
            答案
        """
        logger.info(f"RAG 异步查询: {question}")

        try:
            answer = await self.chain.ainvoke(question)
            # 当前实现通过 StrOutputParser 丢失了原始 usage，使用近似值统计请求级 token
            self._record_usage(question, answer)
            logger.info("✅ RAG 回答生成成功")
            return answer

        except Exception as e:
            logger.error(f"RAG 异步查询失败: {e}")
            raise

    def invoke_with_sources(self, question: str) -> Dict[str, Any]:
        """
        调用 RAG Chain 并返回来源信息

        Args:
            question: 用户问题

        Returns:
            包含答案和来源的字典
        """
        logger.info(f"RAG 查询（含来源）: {question}")

        try:
            # 先检索相关文档
            docs = self.retriever.get_relevant_documents(question)
            logger.info("RAG 来源检索完成（同步）: docs=%s", len(docs))

            # 基于同一批 docs 生成答案，避免二次检索
            answer = self._invoke_from_docs(question, docs)

            # 提取来源信息
            sources = []
            for doc in docs:
                sources.append({
                    "filename": doc.metadata.get("filename", "未知"),
                    "source": doc.metadata.get("source", "未知"),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })

            result = {
                "answer": answer,
                "sources": sources,
                "source_count": len(sources)
            }

            logger.info(f"✅ RAG 回答生成成功，使用了 {len(sources)} 个来源")
            return result

        except Exception as e:
            logger.error(f"RAG 查询（含来源）失败: {e}")
            raise

    async def ainvoke_with_sources(self, question: str) -> Dict[str, Any]:
        """
        异步调用 RAG Chain 并返回来源信息

        Args:
            question: 用户问题

        Returns:
            包含答案和来源的字典
        """
        logger.info(f"RAG 异步查询（含来源）: {question}")

        try:
            # 先检索相关文档（优先使用异步检索，不可用时回退同步）
            if hasattr(self.retriever, "ainvoke"):
                docs = await self.retriever.ainvoke(question)
            else:
                docs = self.retriever.invoke(question)
            logger.info("RAG 来源检索完成（异步）: docs=%s", len(docs))

            # 基于同一批 docs 生成答案，避免二次检索
            answer = await self._ainvoke_from_docs(question, docs)

            # 提取来源信息
            sources = []
            for doc in docs:
                sources.append({
                    "filename": doc.metadata.get("filename", "未知"),
                    "source": doc.metadata.get("source", "未知"),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })

            result = {
                "answer": answer,
                "sources": sources,
                "source_count": len(sources)
            }

            logger.info(f"✅ RAG 回答生成成功，使用了 {len(sources)} 个来源")
            return result

        except Exception as e:
            logger.error(f"RAG 异步查询（含来源）失败: {e}")
            raise
