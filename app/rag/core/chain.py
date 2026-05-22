"""
RAG Chain - 检索增强生成链 (Retrieval-Augmented Generation Chain)

本模块实现了 RAG 系统的核心问答链路，主要功能：
1. 对用户问题进行向量化检索 + 关键词检索（取决于配置的 Retriever）
2. 对结构化问题（如代码字段、枚举类查询）进行补充检索优化
3. 将检索结果格式化为上下文，通过 LLM 生成答案
4. 提供拒答检测与兜底重试机制
5. 记录 Token 使用量

核心流程：
    用户问题 → 检索相关文档 → 格式化上下文 → LLM 生成答案 → 返回答案 + 来源

支持两种调用模式：
- invoke/ainvoke: 纯问答模式（只返回答案字符串）
- invoke_with_sources/ainvoke_with_sources: 带来源模式（返回答案 + 来源文档信息）
"""
from typing import List, Dict, Any, Optional
import re
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import logging

from app.token_usage import record_llm_token_usage

logger = logging.getLogger(__name__)


class RAGChain:
    """
    RAG 问答链

    职责：
    1. 协调 Retriever 与 LLM，构建完整的 RAG 流程
    2. 对特殊类型问题（结构化代码查询）进行优化
    3. 管理 Prompt 模板与上下文格式化
    4. 提供拒答兜底重试机制

    Attributes:
        llm: 语言模型实例（如 ChatOpenAI）
        retriever: 文档检索器（如 ChromaRetriever、HybridRetriever）
        prompt_template: 系统提示词模板
        chain: 构建好的 LangChain Runnable 链
    """

    # =========================================================================
    # 类常量定义
    # =========================================================================

    # 默认 Prompt 模板
    # {context}: 检索到的文档内容（已格式化）
    # {question}: 用户原始问题
    DEFAULT_PROMPT = """你是一个智能问答助手。请基于以下提供的上下文信息来回答用户的问题。

**重要规则**：
1. 只使用上下文中的信息回答问题
2. 如果信息不完整，不要直接拒答；先给出上下文中"可以明确确认"的部分结果，并明确标注"以下为部分信息，可能不完整"
3. 仅当上下文完全没有可用信息时，才明确告诉用户"根据现有资料无法回答该问题"
4. 回答要准确、具体，引用上下文中的关键信息
5. 如果可能，说明信息的来源（如文件名）

上下文信息：
{context}

用户问题：{question}

请提供详细的回答："""

    # 拒答标记词：LLM 输出中包含这些词时，判定为拒绝回答
    # 用于触发兜底重试机制
    REFUSAL_MARKERS = (
        "根据现有资料无法回答该问题",
        "无法回答该问题",
        "没有相关信息",
        "没有包含",
        "未找到相关信息",
        "信息不足",
        "无法确定"
    )

    # 最大上下文文档数：限制传入 LLM 的文档数量，避免超出上下文窗口
    MAX_CONTEXT_DOCS = 12

    # =========================================================================
    # 初始化方法
    # =========================================================================

    def __init__(
        self,
        llm: ChatOpenAI,
        retriever,
        prompt_template: Optional[str] = None
    ):
        """
        初始化 RAG Chain

        Args:
            llm: 语言模型实例（如 ChatOpenAI、ChatOllama 等）
            retriever: 文档检索器，需实现 get_relevant_documents/invoke 或 ainvoke 方法
            prompt_template: 自定义 Prompt 模板，如果不提供则使用默认模板

        Example:
            >>> from langchain_openai import ChatOpenAI
            >>> retriever = vectorstore.as_retriever()
            >>> chain = RAGChain(llm=ChatOpenAI(), retriever=retriever)
        """
        self.llm = llm
        self.retriever = retriever
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT

        # 构建可运行的 LangChain Chain
        self.chain = self._build_chain()

        logger.info("✅ RAG Chain 初始化完成")

    # =========================================================================
    # Token 记录工具方法
    # =========================================================================

    def _record_usage(self, question: str, answer: str):
        """
        记录 Token 使用量（近似统计）

        注意：由于使用了 StrOutputParser，原始 usage 信息已丢失，
        这里使用字符数/3 作为近似的 token 数（中英文平均情况）

        Args:
            question: 用户问题文本
            answer: LLM 生成的答案文本
        """
        record_llm_token_usage(
            prompt_tokens=len(question) // 3,
            completion_tokens=len(answer) // 3,
            model_name=getattr(self.llm, "model_name", "")
        )

    # =========================================================================
    # 消息构建与文档复用方法（用于 _with_sources 系列方法）
    # =========================================================================

    def _build_messages(self, question: str, docs: List[Document]) -> list:
        """
        基于问题与已检索文档构造提示消息

        这个方法用于 _with_sources 系列方法，避免重复检索。
        先调用 Retrieve 获取 docs，然后用此方法构建消息。

        Args:
            question: 用户问题
            docs: 已检索到的文档列表

        Returns:
            格式化后的消息列表（可直接传入 LLM）
        """
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        context = self._format_docs(docs)
        return prompt.format_messages(context=context, question=question)

    def _invoke_from_docs(self, question: str, docs: List[Document]) -> str:
        """
        同步：基于已检索的 docs 直接生成答案（不再二次检索）

        这是 _with_sources 流程的核心方法：
        1. 先使用 _retrieve_with_fallback_sync 获取 docs
        2. 调用此方法生成答案
        3. 避免 Chain 内部再次调用 retriever（节省一次检索）

        Args:
            question: 用户问题
            docs: 已检索到的文档列表

        Returns:
            LLM 生成的答案字符串
        """
        messages = self._build_messages(question, docs)
        response = self.llm.invoke(messages)
        answer = StrOutputParser().invoke(response)
        self._record_usage(question, answer)
        return answer

    async def _ainvoke_from_docs(self, question: str, docs: List[Document]) -> str:
        """
        异步：基于已检索的 docs 直接生成答案（不再二次检索）

        这是 _invoke_from_docs 的异步版本。

        Args:
            question: 用户问题
            docs: 已检索到的文档列表

        Returns:
            LLM 生成的答案字符串
        """
        messages = self._build_messages(question, docs)
        response = await self.llm.ainvoke(messages)
        answer = StrOutputParser().invoke(response)
        self._record_usage(question, answer)
        return answer

    # =========================================================================
    # Chain 构建方法
    # =========================================================================

    def _build_chain(self):
        """
        构建 LangChain Runnable 链

        Chain 结构：
            1. 并行获取 context（通过 retriever + _format_docs）和 question（透传）
            2. 传入 Prompt 模板
            3. 调用 LLM
            4. 解析为字符串输出

        注意：这个 Chain 内部会调用 retriever，所以 invoke/ainvoke 方法
        不需要单独处理检索逻辑。

        Returns:
            构建好的 Runnable Chain
        """
        # 创建 Prompt 模板对象
        prompt = ChatPromptTemplate.from_template(self.prompt_template)

        # 构建 Chain：使用 | 操作符连接各步骤
        chain = (
            {
                # context: 先调用 retriever 获取文档，再格式化为字符串
                "context": self.retriever | self._format_docs,
                # question: 透传原始输入
                "question": RunnablePassthrough()
            }
            | prompt           # 填充模板
            | self.llm         # 调用语言模型
            | StrOutputParser()  # 解析输出为字符串
        )

        return chain

    # =========================================================================
    # 文档处理工具方法
    # =========================================================================

    @staticmethod
    def _merge_unique_docs(primary: List[Document], secondary: List[Document], limit: int) -> List[Document]:
        """
        合并两个文档列表，去重后返回指定数量的文档

        去重策略：基于 (source, start_index, content前120字符) 作为唯一键

        Args:
            primary: 主要文档列表（优先级高，排在前面）
            secondary: 次要文档列表（用于补充）
            limit: 返回的最大文档数量

        Returns:
            合并去重后的文档列表（长度 <= limit）
        """
        merged: List[Document] = []
        seen = set()
        # 先遍历 primary，再遍历 secondary，保证优先级
        for doc in primary + secondary:
            metadata = doc.metadata or {}
            # 构建唯一键：source + start_index + content前120字符
            key = (
                metadata.get("source", ""),
                metadata.get("start_index", ""),
                (doc.page_content or "")[:120]
            )
            if key in seen:
                continue  # 跳过重复文档
            seen.add(key)
            merged.append(doc)
            if len(merged) >= limit:
                break  # 达到数量限制，停止
        return merged

    # =========================================================================
    # 结构化查询识别与优化方法
    # =========================================================================

    @staticmethod
    def _is_structured_question(question: str) -> bool:
        """
        判断是否为结构化问题（如代码字段、枚举类查询）

        结构化特征：
        1. 包含点号（如 "User.name"）- 可能是对象字段访问
        2. 包含下划线（如 "STATUS_ACTIVE"）- 可能是常量/枚举
        3. 包含 6 位以上数字（可能是枚举值或错误码）

        Args:
            question: 用户问题文本

        Returns:
            True 表示是结构化问题，可能需要额外检索优化
        """
        q = (question or "").lower()
        # 只保留通用代码特征，不涉及业务关键词
        if "." in q or "_" in q:
            return True
        if re.search(r"\b\d{6,}\b", q):
            return True
        return False

    @staticmethod
    def _build_expanded_queries(question: str) -> List[str]:
        """
        为结构化问题构造补充检索 query

        当用户查询代码字段、枚举值等结构化内容时，
        使用通用扩展词扩大召回范围。

        Args:
            question: 原始用户问题

        Returns:
            扩展查询列表
        """
        q = (question or "").strip()
        if not q:
            return []
        # 使用通用查询扩展，不涉及具体业务词汇
        return [
            f"{q} definition",      # 英文：定义
            f"{q} meaning",         # 英文：含义
            f"{q} value",           # 英文：取值/值
        ]

    # =========================================================================
    # 拒答检测方法
    # =========================================================================

    @classmethod
    def _is_refusal_answer(cls, answer: str) -> bool:
        """
        检测 LLM 输出是否为拒答

        某些情况下即使检索到相关文档，LLM 也可能因为：
        - 上下文信息不完整
        - 缺乏明确答案
        而拒绝回答。

        检测方法：
        1. 移除答案中所有空白字符
        2. 转小写后检查是否包含任何 REFUSAL_MARKERS

        Args:
            answer: LLM 生成的答案

        Returns:
            True 表示检测到拒答
        """
        normalized = re.sub(r"\s+", "", (answer or "").lower())
        return any(marker.lower().replace(" ", "") in normalized for marker in cls.REFUSAL_MARKERS)

    # =========================================================================
    # 检索方法（同步/异步）
    # =========================================================================

    def _retrieve_once_sync(self, query: str) -> List[Document]:
        """
        执行单次同步检索

        兼容多种 Retriever 接口：
        - langchain 风格的 get_relevant_documents
        - LCEL 风格的 invoke

        Args:
            query: 查询字符串

        Returns:
            检索到的文档列表
        """
        if hasattr(self.retriever, "get_relevant_documents"):
            return self.retriever.get_relevant_documents(query)
        if hasattr(self.retriever, "invoke"):
            docs = self.retriever.invoke(query)
            return docs if isinstance(docs, list) else []
        return []

    async def _retrieve_once_async(self, query: str) -> List[Document]:
        """
        执行单次异步检索

        兼容多种 Retriever 异步接口：
        - ainvoke（LCEL 风格，推荐）
        - aget_relevant_documents（langchain 风格）
        - invoke（同步接口的异步包装）

        Args:
            query: 查询字符串

        Returns:
            检索到的文档列表
        """
        if hasattr(self.retriever, "ainvoke"):
            try:
                docs = await self.retriever.ainvoke(query)
                return docs if isinstance(docs, list) else []
            except KeyError as e:
                # 兼容部分 LangChain 版本在 retriever.ainvoke 回调序列化阶段抛出 KeyError('_type')
                if str(e) == "'_type'" and hasattr(self.retriever, "aget_relevant_documents"):
                    logger.warning("Retriever.ainvoke 触发 KeyError('_type')，降级为 aget_relevant_documents")
                    return await self.retriever.aget_relevant_documents(query)
                if str(e) == "'_type'" and hasattr(self.retriever, "get_relevant_documents"):
                    logger.warning("Retriever.ainvoke 触发 KeyError('_type')，降级为同步检索兜底")
                    return self.retriever.get_relevant_documents(query)
                raise
        if hasattr(self.retriever, "aget_relevant_documents"):
            return await self.retriever.aget_relevant_documents(query)
        if hasattr(self.retriever, "invoke"):
            docs = self.retriever.invoke(query)
            return docs if isinstance(docs, list) else []
        return []

    def _retrieve_with_fallback_sync(self, question: str, force_boost: bool = False) -> List[Document]:
        """
        同步检索，支持补充检索兜底（Fallback）

        逻辑：
        1. 先用原始问题检索
        2. 如果是结构化问题且结果少，则用扩展查询补充检索
        3. 合并去重后返回

        Args:
            question: 用户问题
            force_boost: 强制启用补充检索（不理会结构化检测）

        Returns:
            检索到的文档列表（已合并去重，长度 <= MAX_CONTEXT_DOCS）
        """
        # 第一次检索：使用原始问题
        docs = self._retrieve_once_sync(question)

        # 判断是否需要进行补充检索
        should_boost = force_boost or (self._is_structured_question(question) and len(docs) < self.MAX_CONTEXT_DOCS)
        if not should_boost:
            return docs

        # 补充检索：使用扩展查询
        expanded_docs: List[Document] = docs
        for query in self._build_expanded_queries(question):
            extra = self._retrieve_once_sync(query)
            expanded_docs = self._merge_unique_docs(expanded_docs, extra, self.MAX_CONTEXT_DOCS)
            if len(expanded_docs) >= self.MAX_CONTEXT_DOCS:
                break  # 已达到上限，停止补充

        logger.info("RAG 补充检索（同步）: base_docs=%s, merged_docs=%s", len(docs), len(expanded_docs))
        return expanded_docs

    async def _retrieve_with_fallback_async(self, question: str, force_boost: bool = False) -> List[Document]:
        """
        异步检索，支持补充检索兜底

        这是 _retrieve_with_fallback_sync 的异步版本，逻辑相同。

        Args:
            question: 用户问题
            force_boost: 强制启用补充检索

        Returns:
            检索到的文档列表
        """
        # 第一次检索
        docs = await self._retrieve_once_async(question)

        # 判断是否补充检索
        should_boost = force_boost or (self._is_structured_question(question) and len(docs) < self.MAX_CONTEXT_DOCS)
        if not should_boost:
            return docs

        # 补充检索
        expanded_docs: List[Document] = docs
        for query in self._build_expanded_queries(question):
            extra = await self._retrieve_once_async(query)
            expanded_docs = self._merge_unique_docs(expanded_docs, extra, self.MAX_CONTEXT_DOCS)
            if len(expanded_docs) >= self.MAX_CONTEXT_DOCS:
                break

        logger.info("RAG 补充检索（异步）: base_docs=%s, merged_docs=%s", len(docs), len(expanded_docs))
        return expanded_docs

    # =========================================================================
    # 文档格式化方法
    # =========================================================================

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """
        将文档列表格式化为上下文字符串

        格式示例：
            【文档 1】（来源: README.md）
            文档内容...

            【文档 2】（来源: api.md）
            文档内容...

        Args:
            docs: 文档列表

        Returns:
            格式化后的上下文字符串，用于填充 Prompt
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

    # =========================================================================
    # 公开调用方法
    # =========================================================================

    def invoke(self, question: str) -> str:
        """
        同步调用 RAG Chain（纯问答模式）

        特点：
        - 只返回答案字符串
        - 内部会自动调用 retriever 进行检索
        - 不返回来源信息

        适用场景：
        - 只需要答案，不关心来源
        - 简单问答接口

        Args:
            question: 用户问题

        Returns:
            LLM 生成的答案字符串

        Note:
            此方法直接使用 self.chain.invoke，会触发链内的检索逻辑。
            如需控制检索过程，请使用 invoke_with_sources。
        """
        logger.info(f"RAG 查询: {question}")

        try:
            answer = self.chain.invoke(question)
            # StrOutputParser 丢失了原始 usage，使用近似值统计
            self._record_usage(question, answer)
            logger.info("✅ RAG 回答生成成功")
            return answer

        except Exception as e:
            logger.error(f"RAG 查询失败: {e}")
            raise

    async def ainvoke(self, question: str) -> str:
        """
        异步调用 RAG Chain（纯问答模式）

        这是 invoke 的异步版本，功能相同。

        Args:
            question: 用户问题

        Returns:
            LLM 生成的答案字符串
        """
        logger.info(f"RAG 异步查询: {question}")

        try:
            answer = await self.chain.ainvoke(question)
            self._record_usage(question, answer)
            logger.info("✅ RAG 回答生成成功")
            return answer

        except Exception as e:
            logger.error(f"RAG 异步查询失败: {e}")
            raise

    def invoke_with_sources(self, question: str) -> Dict[str, Any]:
        """
        同步调用 RAG Chain 并返回来源信息（推荐）

        完整流程：
        1. 使用 _retrieve_with_fallback_sync 检索文档（支持补充检索）
        2. 使用 _invoke_from_docs 生成答案（避免二次检索）
        3. 检测是否为拒答，如果是则重试
        4. 按文件去重，提取来源信息

        Args:
            question: 用户问题

        Returns:
            包含以下字段的字典：
            {
                "answer": str,          # 答案文本
                "sources": List[dict],  # 来源列表，每项包含 filename, source, content_preview
                "source_count": int     # 来源数量
            }

        Example:
            >>> result = chain.invoke_with_sources("什么是枚举类？")
            >>> print(result["answer"])
            >>> for s in result["sources"]:
            >>>     print(f"来源: {s['filename']}")
        """
        logger.info(f"RAG 查询（含来源）: {question}")

        try:
            # Step 1: 检索文档（结构化问题会自动补充检索）
            docs = self._retrieve_with_fallback_sync(question)
            logger.info("RAG 来源检索完成（同步）: docs=%s", len(docs))

            # Step 2: 基于同一批 docs 生成答案（避免 chain 内部二次检索）
            answer = self._invoke_from_docs(question, docs)

            # Step 3: 拒答检测与兜底重试
            # 如果 LLM 拒答，但确实有检索到文档，尝试强制补充检索
            if self._is_refusal_answer(answer) and docs:
                retry_docs = self._retrieve_with_fallback_sync(question, force_boost=True)
                if len(retry_docs) > len(docs):  # 只有补充到新文档才重试
                    retry_answer = self._invoke_from_docs(question, retry_docs)
                    if not self._is_refusal_answer(retry_answer):
                        docs = retry_docs
                        answer = retry_answer
                        logger.info("RAG 同步拒答兜底重试成功: docs=%s", len(docs))

            # Step 5: 提取来源信息（按文件去重，每个文件只保留第一个匹配块）
            sources = []
            seen_files = set()
            for doc in docs:
                filename = doc.metadata.get("filename", "未知")
                source = doc.metadata.get("source", "未知")

                # 使用 source（完整路径）作为去重键
                if source not in seen_files:
                    seen_files.add(source)
                    sources.append({
                        "filename": filename,
                        "source": source,
                        "content_preview": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
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

        这是 invoke_with_sources 的异步版本，功能与流程完全相同。

        Args:
            question: 用户问题

        Returns:
            包含答案和来源的字典（格式同 invoke_with_sources）
        """
        logger.info(f"RAG 异步查询（含来源）: {question}")

        try:
            # Step 1: 异步检索
            docs = await self._retrieve_with_fallback_async(question)

            logger.info("RAG 来源检索完成（异步）: docs=%s", len(docs))

            # Step 2: 异步生成答案
            answer = await self._ainvoke_from_docs(question, docs)

            # Step 3: 拒答检测与异步兜底重试
            if self._is_refusal_answer(answer) and docs:
                retry_docs = await self._retrieve_with_fallback_async(question, force_boost=True)

                if len(retry_docs) > len(docs):
                    retry_answer = await self._ainvoke_from_docs(question, retry_docs)
                    if not self._is_refusal_answer(retry_answer):
                        docs = retry_docs
                        answer = retry_answer
                        logger.info("RAG 异步拒答兜底重试成功: docs=%s", len(docs))

            # Step 4: 提取来源信息
            sources = []
            seen_files = set()
            for doc in docs:
                filename = doc.metadata.get("filename", "未知")
                source = doc.metadata.get("source", "未知")

                if source not in seen_files:
                    seen_files.add(source)
                    sources.append({
                        "filename": filename,
                        "source": source,
                        "content_preview": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                    })

            result = {
                "answer": answer,
                "sources": sources,
                "source_count": len(sources)
            }

            logger.info(f"✅ RAG 回答生成成功，使用了 {len(sources)} 个来源")
            return result

        except Exception as e:
            logger.exception(f"RAG 异步查询（含来源）失败: {e}")
            raise
