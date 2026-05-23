"""
多模型 + ES 混合检索器 - RRF 三路融合版本
支持 BGE、代码模型(UniXcoder/CodeBERT)、ES 三路并行检索，使用 RRF 算法融合结果
"""
from typing import List, Optional, Any, Dict
from collections import defaultdict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
import logging
import asyncio

logger = logging.getLogger(__name__)


class MultiModelHybridRetriever(BaseRetriever):
    """
    多模型 Hybrid 检索器 - RRF 三路融合
    
    三路并行召回：
    1. 文档向量检索器（BGE等文档类模型）
    2. 代码向量检索器（UniXcoder/CodeBERT等代码模型）
    3. ES 关键词检索
    
    使用 RRF (Reciprocal Rank Fusion) 算法融合，无需分数归一化
    """
    
    # 三路检索器配置
    doc_retriever: Optional[Any] = Field(None, description="文档向量检索器(BGE等)")
    code_retriever: Optional[Any] = Field(None, description="代码向量检索器(UniXcoder/CodeBERT等)")
    es_retriever: Optional[Any] = Field(None, description="ES关键词检索器")
    
    # 向后兼容的旧字段名
    bge_retriever: Optional[Any] = Field(None, description="[兼容]BGE向量检索器，映射到doc_retriever")
    codebert_retriever: Optional[Any] = Field(None, description="[兼容]CodeBERT向量检索器，映射到code_retriever")
    
    def __init__(self, **data):
        """初始化，处理向后兼容的字段名"""
        # 处理旧字段名映射
        if 'bge_retriever' in data and data['bge_retriever'] is not None:
            data['doc_retriever'] = data.pop('bge_retriever')
        if 'codebert_retriever' in data and data['codebert_retriever'] is not None:
            data['code_retriever'] = data.pop('codebert_retriever')
        super().__init__(**data)
    
    # RRF 参数
    rrf_k: int = Field(60, description="RRF 算法中的常数k，Google推荐60")
    k: int = Field(5, description="返回结果数")
    
    # 每路检索的 top_k（内部使用，可配置）
    per_retriever_k: int = Field(20, description="每路检索器返回的结果数")
    
    class Config:
        arbitrary_types_allowed = True

    def _doc_key(self, doc: Document) -> str:
        """生成文档唯一标识键"""
        metadata = doc.metadata or {}
        source = metadata.get("source", "")
        filename = metadata.get("filename", "")
        start_index = metadata.get("start_index", "")
        # 使用内容前200字符的哈希作为指纹
        content_preview = doc.page_content[:200].strip() if doc.page_content else ""
        return f"{source}|{filename}|{start_index}|{hash(content_preview)}"

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        执行三路并行检索，使用 RRF 融合结果
        
        Args:
            query: 查询文本
            
        Returns:
            RRF 融合后的文档列表
        """
        logger.info(f"RRF 三路检索开始: query={query[:50]}...")
        
        # 1. 三路并行检索
        results_by_source: Dict[str, List[Document]] = {}
        
        def _invoke_retriever(retriever, query_str):
            """兼容不同检索器接口"""
            if retriever is None:
                return []
            try:
                # 优先使用 invoke (LangChain 0.1.46+ 推荐)
                if hasattr(retriever, 'invoke'):
                    result = retriever.invoke(query_str)
                    if isinstance(result, list):
                        return result
                    return [result] if result else []
                # 回退到 get_relevant_documents
                elif hasattr(retriever, 'get_relevant_documents'):
                    return retriever.get_relevant_documents(query_str)
                else:
                    logger.warning(f"检索器没有可用的调用方法")
                    return []
            except Exception as e:
                raise e
        
        # 文档检索器 (BGE)
        if self.doc_retriever:
            try:
                doc_results = _invoke_retriever(self.doc_retriever, query)
                results_by_source['doc'] = doc_results[:self.per_retriever_k]
                logger.info(f"  文档检索器: {len(results_by_source['doc'])} 条")
            except Exception as e:
                logger.warning(f"文档检索失败: {e}")
                results_by_source['doc'] = []
        else:
            results_by_source['doc'] = []
        
        # 代码检索器 (UniXcoder/CodeBERT)
        if self.code_retriever:
            try:
                code_results = _invoke_retriever(self.code_retriever, query)
                results_by_source['code'] = code_results[:self.per_retriever_k]
                logger.info(f"  代码检索器: {len(results_by_source['code'])} 条")
            except Exception as e:
                logger.warning(f"代码检索失败: {e}")
                results_by_source['code'] = []
        else:
            results_by_source['code'] = []
        
        # ES 检索
        if self.es_retriever:
            try:
                es_results = _invoke_retriever(self.es_retriever, query)
                results_by_source['es'] = es_results[:self.per_retriever_k]
                logger.info(f"  ES检索: {len(results_by_source['es'])} 条")
            except Exception as e:
                logger.warning(f"ES 检索失败: {e}")
                results_by_source['es'] = []
        else:
            results_by_source['es'] = []
        
        # 2. RRF 融合
        final_results = self._rrf_fusion(results_by_source)
        
        logger.info(f"RRF 融合完成: 返回 {len(final_results)} 条结果")
        return final_results

    def _rrf_fusion(self, results_by_source: Dict[str, List[Document]]) -> List[Document]:
        """
        RRF (Reciprocal Rank Fusion) 融合算法
        
        公式: score = Σ 1/(k + rank)
        
        其中 k 是常数（默认60），rank 是文档在该源中的排名（从1开始）
        
        Args:
            results_by_source: 各检索源的结果列表
            
        Returns:
            按 RRF 分数排序的文档列表
        """
        # 文档 -> RRF 分数 的映射
        rrf_scores: Dict[str, float] = defaultdict(float)
        # 文档键 -> Document 的映射
        doc_map: Dict[str, Document] = {}
        # 记录每个文档的来源（用于调试）
        doc_sources: Dict[str, List[str]] = defaultdict(list)
        
        # 对每个源的结果计算 RRF 分数
        for source_name, docs in results_by_source.items():
            for rank, doc in enumerate(docs, start=1):  # rank 从1开始
                doc_key = self._doc_key(doc)
                
                # RRF 公式: 1/(k + rank)
                rrf_score = 1.0 / (self.rrf_k + rank)
                rrf_scores[doc_key] += rrf_score
                
                # 保存文档和来源信息
                if doc_key not in doc_map:
                    doc_map[doc_key] = doc
                    # 复制文档避免修改原始对象
                    doc_map[doc_key] = Document(
                        page_content=doc.page_content,
                        metadata=dict(doc.metadata) if doc.metadata else {}
                    )
                
                doc_sources[doc_key].append(f"{source_name}#{rank}")
                
                logger.debug(f"  RRF: {doc_key[:50]}... from {source_name} rank={rank} "
                           f"score={rrf_score:.4f} cumsum={rrf_scores[doc_key]:.4f}")
        
        # 添加 RRF 分数和来源信息到文档 metadata
        for doc_key, doc in doc_map.items():
            doc.metadata['_rrf_score'] = rrf_scores[doc_key]
            doc.metadata['_sources'] = doc_sources[doc_key]
        
        # 按 RRF 分数降序排序
        sorted_docs = sorted(
            doc_map.values(),
            key=lambda x: x.metadata.get('_rrf_score', 0),
            reverse=True
        )
        
        # 去重（RRF 已经将多源结果合并，这里只是确保返回指定数量）
        unique_results = sorted_docs[:self.k]
        
        # 记录前几名的详细信息
        for i, doc in enumerate(unique_results[:5], 1):
            score = doc.metadata.get('_rrf_score', 0)
            sources = doc.metadata.get('_sources', [])
            source_file = doc.metadata.get('source', 'unknown')
            filename = doc.metadata.get('filename', 'unknown')
            content_preview = doc.page_content[:80].replace('\n', ' ') if doc.page_content else ''
            
            # 解析来源：哪些模型/ES，各自排名
            source_details = []
            for src in sources:
                # src 格式: "doc#3" 或 "es#1"
                if '#' in src:
                    src_name, rank = src.split('#')
                    source_details.append(f"{src_name}(#{rank})")
                else:
                    source_details.append(src)
            
            logger.info(f"  TOP-{i}: RRF分数={score:.4f}, 来源={source_details}")
            logger.info(f"         文件={filename}, 路径={source_file}")
            logger.info(f"         内容预览={content_preview}...")
        
        # 如果结果少于5条，记录所有结果
        if len(unique_results) <= 5:
            for i, doc in enumerate(unique_results, 1):
                score = doc.metadata.get('_rrf_score', 0)
                sources = doc.metadata.get('_sources', [])
                logger.info(f"  结果-{i}: RRF分数={score:.4f}, 来源={sources}")
        
        return unique_results

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """
        异步版本 - 三路并发检索
        """
        logger.info(f"RRF 三路检索开始 (async): query={query[:50]}...")
        
        # 1. 三路并发检索
        tasks = []
        
        if self.doc_retriever:
            tasks.append(self._async_retrieve('doc', self.doc_retriever, query))
        else:
            tasks.append(asyncio.sleep(0, result=('doc', [])))
            
        if self.code_retriever:
            tasks.append(self._async_retrieve('code', self.code_retriever, query))
        else:
            tasks.append(asyncio.sleep(0, result=('code', [])))
            
        if self.es_retriever:
            tasks.append(self._async_retrieve('es', self.es_retriever, query))
        else:
            tasks.append(asyncio.sleep(0, result=('es', [])))
        
        # 等待所有结果
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 收集结果
        results_by_source: Dict[str, List[Document]] = {
            'doc': [],
            'code': [],
            'es': []
        }
        
        for result in results_list:
            if isinstance(result, Exception):
                logger.warning(f"检索任务失败: {result}")
                continue
            source_name, docs = result
            results_by_source[source_name] = docs[:self.per_retriever_k]
            if source_name == 'doc':
                logger.info(f"  文档检索器: {len(results_by_source[source_name])} 条")
            elif source_name == 'code':
                logger.info(f"  代码检索器: {len(results_by_source[source_name])} 条")
            else:
                logger.info(f"  {source_name}检索: {len(results_by_source[source_name])} 条")
        
        # 2. RRF 融合
        final_results = self._rrf_fusion(results_by_source)
        
        logger.info(f"RRF 融合完成: 返回 {len(final_results)} 条结果")
        return final_results

    async def _async_retrieve(self, source_name: str, retriever: Any, query: str):
        """带来源标识的异步检索"""
        try:
            if retriever is None:
                return (source_name, [])
            # 优先使用 ainvoke (LangChain 0.1.46+ 推荐)
            if hasattr(retriever, 'ainvoke'):
                result = await retriever.ainvoke(query)
                # ainvoke 可能返回 Document 列表或其他格式
                if isinstance(result, list):
                    docs = result
                else:
                    docs = [result] if result else []
            elif hasattr(retriever, 'aget_relevant_documents'):
                docs = await retriever.aget_relevant_documents(query)
            else:
                # 回退到同步方法
                docs = await asyncio.to_thread(retriever.get_relevant_documents, query)
            return (source_name, docs)
        except Exception as e:
            logger.warning(f"{source_name} 异步检索失败: {e}")
            return (source_name, [])

    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        return {
            "type": "MultiModelHybridRetriever",
            "algorithm": "RRF",
            "rrf_k": self.rrf_k,
            "k": self.k,
            "per_retriever_k": self.per_retriever_k,
            "retrievers": {
                "doc": self.doc_retriever is not None,
                "code": self.code_retriever is not None,
                "es": self.es_retriever is not None
            }
        }
