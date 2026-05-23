"""
多模型检索器
支持从多个向量库检索并合并结果
"""
from typing import List, Dict, Any
from langchain_core.documents import Document
import logging
import asyncio

logger = logging.getLogger(__name__)


class MultiModelRetriever:
    """
    多模型检索器

    从多个向量库检索结果，合并后返回
    支持加权、去重、重排序
    """

    def __init__(
        self,
        retrievers: List[Dict[str, Any]],
        k: int = 5,
        dedup_threshold: float = 0.9
    ):
        """
        初始化多模型检索器

        Args:
            retrievers: 检索器配置列表，每项包含 retriever、weight、model_config
            k: 最终结果数量
            dedup_threshold: 去重阈值（内容相似度）
        """
        self.retrievers = retrievers
        self.k = k
        self.dedup_threshold = dedup_threshold

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        执行多路检索并合并结果

        Args:
            query: 查询文本

        Returns:
            合并后的文档列表
        """
        logger.info(f"多模型检索: {query[:50]}...")
        
        all_results = []
        
        # 从每个检索器获取结果
        for retriever_config in self.retrievers:
            retriever = retriever_config["retriever"]
            weight = retriever_config.get("weight", 1.0)
            model_config = retriever_config.get("model_config")
            
            try:
                results = retriever.get_relevant_documents(query)
                logger.info(
                    f"  {model_config.model_name if model_config else 'unknown'}: "
                    f"{len(results)} 条 (weight={weight})"
                )
                
                # 添加权重标记
                for doc in results:
                    base_score = doc.metadata.get("score", 0.5)
                    doc.metadata["_weighted_score"] = base_score * weight
                    doc.metadata["_model"] = model_config.model_name if model_config else "unknown"
                    all_results.append(doc)
                    
            except Exception as e:
                logger.warning(f"检索器失败: {e}")
        
        logger.info(f"原始结果总数: {len(all_results)}")
        
        # 去重
        unique_results = self._deduplicate(all_results)
        logger.info(f"去重后结果: {len(unique_results)}")
        
        # 按加权分数排序
        unique_results.sort(
            key=lambda x: x.metadata.get("_weighted_score", 0),
            reverse=True
        )
        
        return unique_results[:self.k]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """异步版本"""
        logger.info(f"多模型检索 (async): {query[:50]}...")
        
        # 并发执行所有检索
        tasks = []
        for retriever_config in self.retrievers:
            retriever = retriever_config["retriever"]
            task = asyncio.create_task(
                self._retrieve_with_config(retriever, retriever_config, query)
            )
            tasks.append(task)
        
        # 等待所有结果
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 合并结果
        all_results = []
        for results in results_list:
            if isinstance(results, list):
                all_results.extend(results)
            else:
                logger.warning(f"检索失败: {results}")
        
        logger.info(f"原始结果总数: {len(all_results)}")
        
        # 去重和排序
        unique_results = self._deduplicate(all_results)
        unique_results.sort(
            key=lambda x: x.metadata.get("_weighted_score", 0),
            reverse=True
        )
        
        return unique_results[:self.k]

    async def _retrieve_with_config(self, retriever, config, query: str) -> List[Document]:
        """带配置的异步检索"""
        weight = config.get("weight", 1.0)
        model_config = config.get("model_config")
        
        try:
            results = retriever.get_relevant_documents(query)
            for doc in results:
                base_score = doc.metadata.get("score", 0.5)
                doc.metadata["_weighted_score"] = base_score * weight
                doc.metadata["_model"] = model_config.model_name if model_config else "unknown"
            return results
        except Exception as e:
            logger.warning(f"检索失败: {e}")
            return []

    def _deduplicate(self, documents: List[Document]) -> List[Document]:
        """
        基于内容相似度去重
        
        Args:
            documents: 文档列表

        Returns:
            去重后的文档列表
        """
        unique_docs = []
        seen_hashes = set()
        
        for doc in documents:
            # 生成内容指纹（前200字符的哈希）
            content_preview = doc.page_content[:200].strip()
            
            # 使用 source + 内容预览去重
            source = doc.metadata.get("source", "")
            key = f"{source}_{hash(content_preview)}"
            
            # 检查是否已存在
            is_duplicate = False
            for seen_key in seen_hashes:
                # 相同来源且内容相似
                if seen_key.startswith(source):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_hashes.add(key)
                unique_docs.append(doc)
        
        return unique_docs

    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        return {
            "retriever_count": len(self.retrievers),
            "k": self.k,
            "dedup_threshold": self.dedup_threshold,
            "models": [
                {
                    "model": r.get("model_config", {}).model_name if r.get("model_config") else "unknown",
                    "weight": r.get("weight", 1.0)
                }
                for r in self.retrievers
            ]
        }
