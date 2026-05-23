"""
RAG 模块日志重构指南

## 重构目标
1. 统一日志格式，使用结构化日志
2. 移除所有 print 语句
3. 添加性能监控
4. 统一使用 % 格式化（性能优化）
5. 完善异常日志，包含堆栈信息

## 主要变更

### 1. 新增工具模块
- app/rag/utils/logging_utils.py - 日志工具函数
- app/rag/utils/logging_config.py - 日志配置

### 2. 日志级别规范
- DEBUG: 详细调试信息，仅在 RAG_VERBOSE=true 时输出
- INFO: 正常流程信息，用户可见的关键步骤
- WARNING: 降级处理、配置异常
- ERROR: 操作失败

### 3. 日志格式规范
```
%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s
```

### 4. 性能日志
使用 timed 上下文管理器/装饰器：
```python
from app.rag.utils.logging_utils import timed

with timed("文档加载", logger):
    docs = loader.load()
```

### 5. 结构化日志
```python
from app.rag.utils.logging_utils import log_structured, RAGLogEvent

log_structured(
    logger, logging.INFO, 
    RAGLogEvent.RETRIEVE_COMPLETE,
    "检索完成",
    query=question[:100],
    results_count=len(docs)
)
```

## 文件变更列表

### 高优先级（先改）
1. system.py - 系统主控制器
2. chain.py - RAG 问答链
3. multi_model_hybrid_retriever.py - 多模型混合检索

### 中优先级
4. embeddings.py - 嵌入模型
5. code_embeddings.py - UniXcoder
6. vectorstore.py - 向量存储

### 低优先级
7. document_loader.py - 文档加载
8. splitter.py - 文本分割
9. es_keyword_retriever.py - ES 检索
10. document_registry.py - 文档注册表

## 测试验证

重启应用后，日志应该：
1. 不再出现 print 输出
2. 格式统一，包含时间、级别、模块名
3. 性能关键路径有 [PERF] 标记
4. 异常有完整的堆栈跟踪
"""

# 以下是各个文件的具体修改示例

SYSTEM_PY_CHANGES = """
## system.py 修改

### 导入修改
```python
# 新增
from app.rag.utils.logging_utils import get_logger, timed, log_structured, RAGLogEvent

# 替换 logger = logging.getLogger(__name__)
logger = get_logger(__name__)
```

### 初始化方法修改
```python
# 旧代码
logger.info("=" * 60)
logger.info("初始化 RAG 系统...")
logger.info("=" * 60)

# 新代码
log_structured(logger, logging.INFO, RAGLogEvent.SYSTEM_INIT_START, "开始初始化 RAG 系统")
```

### 移除 print 语句
```python
# 旧代码 (line 803, 807)
print(f"🧭 [RAG Router] 智能路由：...")

# 新代码
logger.info("[RAG Router] 智能路由：...")
```

### 添加性能监控
```python
# 旧代码
embedding_manager = EmbeddingManager(...)

# 新代码
with timed("初始化嵌入模型", logger):
    embedding_manager = EmbeddingManager(...)
```
"""

CHAIN_PY_CHANGES = """
## chain.py 修改

### 检索性能监控
```python
async def _retrieve_once_async(self, query: str) -> List[Document]:
    with timed(f"检索: {query[:50]}", logger, level=logging.DEBUG):
        # 原有逻辑
```

### 统一格式化
```python
# 旧代码
logger.info(f"RAG 异步查询: {question}")

# 新代码
logger.info("RAG 异步查询: %s", question)
```
"""

RETRIEVER_PY_CHANGES = """
## multi_model_hybrid_retriever.py 修改

### RRF 融合性能监控
```python
def _rrf_fusion(self, results_by_source: Dict[str, List[Document]]) -> List[Document]:
    with timed("RRF 融合", logger, level=logging.DEBUG):
        # 原有逻辑
```

### 结构化检索日志
```python
log_structured(
    logger, logging.INFO,
    RAGLogEvent.RETRIEVE_RRF_FUSION,
    "RRF 融合完成",
    doc_results=len(results_by_source.get('doc', [])),
    code_results=len(results_by_source.get('code', [])),
    es_results=len(results_by_source.get('es', [])),
    final_results=len(unique_results)
)
```
"""
