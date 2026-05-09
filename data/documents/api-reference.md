# API 参考文档

## 基础信息

- **Base URL**: `http://localhost:8000`
- **Content-Type**: `application/json`
- **认证**: 暂无（未来版本将添加）

## 对话接口

### POST /chat

发送消息进行对话。

**请求体**:
```json
{
  "message": "用户消息内容",
  "session_id": "会话ID（可选，默认为default）"
}
```

**响应**:
```json
{
  "response": "AI的回复内容",
  "session_id": "会话ID"
}
```

**示例**:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "今天天气怎么样？", "session_id": "user123"}'
```

### POST /chat/stream

流式对话接口，使用 Server-Sent Events (SSE) 逐字返回响应。

**请求体**: 与 `/chat` 相同

**响应**: SSE 事件流
```
data: {"content": "你"}
data: {"content": "好"}
data: [DONE]
```

**示例**:
```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "讲个故事", "session_id": "user123"}' \
  --no-buffer
```

## 历史管理接口

### GET /history/{session_id}

获取指定会话的历史记录。

**路径参数**:
- `session_id`: 会话ID

**响应**:
```json
{
  "session_id": "user123",
  "history": [
    {
      "type": "human",
      "content": "你好"
    },
    {
      "type": "ai",
      "content": "你好！有什么我可以帮助你的吗？"
    }
  ]
}
```

### DELETE /history/{session_id}

清除指定会话的历史记录。

**路径参数**:
- `session_id`: 会话ID

**响应**:
```json
{
  "message": "会话 user123 的历史记录已清除"
}
```

## RAG 知识库接口

### POST /rag/query

基于知识库回答问题。

**请求体**:
```json
{
  "question": "用户问题",
  "with_sources": true
}
```

**响应**:
```json
{
  "answer": "基于知识库的答案",
  "sources": [
    {
      "filename": "product-guide.md",
      "source": "/path/to/product-guide.md",
      "content_preview": "相关内容片段..."
    }
  ],
  "source_count": 1
}
```

**注意**: 需要先启用 RAG 功能（`RAG_ENABLED=true`）。

### POST /rag/rebuild

重建知识库索引。

**响应**:
```json
{
  "message": "知识库重建完成",
  "status": "success",
  "original_documents": 5,
  "split_documents": 45,
  "indexed_documents": 45,
  "collection_name": "langchain_docs"
}
```

**使用场景**: 
- 添加新文档后
- 删除或更新文档后
- 向量数据库损坏时

### GET /rag/stats

获取 RAG 系统统计信息。

**响应**:
```json
{
  "initialized": true,
  "enabled": true,
  "embedding_model": "text-embedding-3-small",
  "vector_store": "chroma",
  "collection_name": "langchain_docs",
  "document_count": 45,
  "documents_path": "/path/to/documents",
  "vectordb_path": "/path/to/chroma",
  "config": {
    "top_k": 4,
    "search_type": "similarity",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "rebuild_on_startup": true
  }
}
```

## 工具管理接口

### GET /tools

列出所有可用的工具。

**响应**:
```json
{
  "enabled": true,
  "total": 9,
  "tools": [
    {
      "name": "get_current_date_tool",
      "description": "获取当前日期和时间"
    },
    {
      "name": "search_web",
      "description": "使用 DuckDuckGo 搜索网络"
    }
  ]
}
```

## 会话管理接口

### GET /sessions

列出所有会话ID。

**响应**:
```json
{
  "total": 3,
  "sessions": ["user123", "user456", "default"]
}
```

### GET /database/stats

获取数据库统计信息。

**响应**:
```json
{
  "total_sessions": 3,
  "total_checkpoints": 156,
  "database_size_mb": 2.45,
  "sessions": [
    {
      "session_id": "user123",
      "checkpoint_count": 50,
      "last_updated": "2024-01-15T10:30:00"
    }
  ]
}
```

### GET /database/sessions/{session_id}

获取指定会话的详细信息。

**响应**:
```json
{
  "session_id": "user123",
  "checkpoint_count": 50,
  "checkpoints": [
    {
      "checkpoint_id": "abc123",
      "created_at": "2024-01-15T10:30:00",
      "message_count": 20
    }
  ]
}
```

## Token 统计接口

### GET /stats/tokens/{session_id}

获取指定会话的 Token 使用统计。

**响应**:
```json
{
  "session_id": "user123",
  "request_count": 25,
  "total_tokens": 15000,
  "total_cost_usd": 0.045,
  "by_model": {
    "gpt-4o-mini": {
      "request_count": 25,
      "prompt_tokens": 10000,
      "completion_tokens": 5000,
      "total_tokens": 15000,
      "cost_usd": 0.045
    }
  }
}
```

### GET /stats/tokens/daily

获取每日 Token 使用汇总。

**查询参数**:
- `date`: 日期（YYYY-MM-DD），默认为今天

**响应**:
```json
{
  "date": "2024-01-15",
  "unique_sessions": 5,
  "total_requests": 100,
  "total_tokens": 50000,
  "total_cost_usd": 0.15
}
```

### GET /stats/tokens/monthly

获取每月 Token 使用汇总。

**查询参数**:
- `year_month`: 年月（YYYY-MM），默认为当月

**响应**:
```json
{
  "year_month": "2024-01",
  "unique_sessions": 20,
  "total_requests": 1000,
  "total_tokens": 500000,
  "total_cost_usd": 1.50
}
```

## 系统接口

### GET /health

健康检查接口。

**响应**:
```json
{
  "status": "healthy",
  "agent_initialized": true,
  "storage_type": "SQLite",
  "tools_enabled": true,
  "tools_count": 9,
  "langfuse_enabled": false,
  "langfuse_sample_rate": 0.0
}
```

### GET /

获取 API 概览和功能列表。

**响应**:
```json
{
  "message": "LangChain对话Agent API (with Tool Calling + RAG)",
  "storage": "SQLite Persistent Storage",
  "features": [
    "Tool Calling - 自动调用工具完成任务",
    "Context Compression - 上下文压缩",
    "Persistent Storage - SQLite 持久化",
    "Multi-Session - 多会话管理",
    "Token Statistics - Token使用统计和成本追踪",
    "RAG - 检索增强生成（知识库问答）"
  ],
  "endpoints": {
    "POST /chat": "发送消息进行对话",
    "POST /chat/stream": "流式对话（Server-Sent Events）",
    "...": "..."
  }
}
```

## 错误处理

所有接口在发生错误时都会返回标准错误响应：

```json
{
  "detail": "错误详细信息"
}
```

常见 HTTP 状态码：
- `200`: 成功
- `400`: 请求参数错误
- `404`: 资源不存在
- `500`: 服务器内部错误
- `503`: 服务不可用（如 RAG 未初始化）

## 速率限制

当前版本暂无速率限制。生产环境建议配置反向代理（如 Nginx）实现速率限制。

## WebSocket 支持

当前版本使用 HTTP + SSE 实现流式响应。未来版本可能添加 WebSocket 支持。

## 认证与授权

当前版本未实现认证。如需部署到公网，建议：
1. 使用反向代理添加 API Key 认证
2. 配置 IP 白名单
3. 使用 HTTPS
4. 实现 OAuth 2.0

## API 变更日志

### v1.0.0 (2024-01)
- 初始版本发布
- 基础对话接口
- RAG 知识库接口
- Token 统计接口
