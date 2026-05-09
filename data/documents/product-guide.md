# LangChain 对话 Agent 产品指南

## 产品概述

LangChain 对话 Agent 是一个基于 LangChain 和 LangGraph 构建的智能对话系统，支持多会话管理、工具调用、上下文压缩和 RAG 知识库问答等功能。

## 核心功能

### 1. 智能对话
- 支持多轮对话，自动维护上下文
- 基于 GPT-4o-mini 等先进语言模型
- 支持流式响应，实时输出

### 2. 工具调用
系统内置 9 个实用工具：
- 时间工具：获取当前日期、时间戳、星期
- 网络工具：DuckDuckGo 搜索、网页内容抓取
- 实用工具：随机数生成、天气查询、货币转换、计算器

Agent 会根据用户问题自动选择和调用合适的工具。

### 3. 会话管理
- 基于 SQLite 的持久化存储
- 支持多会话并行，通过 session_id 隔离
- 可查看、清除任意会话的历史记录

### 4. 上下文压缩
提供 4 种压缩策略：
- none：不压缩（默认）
- sliding_window：滑动窗口
- token_limit：Token 限制
- summary：智能摘要

### 5. RAG 知识库问答
- 支持本地文档（TXT、Markdown、PDF）
- 基于 Chroma 向量数据库
- 自动引用来源，提高答案可信度

### 6. 监控和统计
- Token 使用统计和成本追踪
- Langfuse 云端监控（可选）
- 实时健康检查

## 使用场景

### 场景 1：客户服务
部署为客服机器人，回答常见问题。通过 RAG 系统加载产品手册和 FAQ，提供准确的产品信息。

### 场景 2：技术支持
作为技术助手，帮助用户解决技术问题。可以调用搜索工具查找最新解决方案，也可以从内部技术文档中检索信息。

### 场景 3：知识管理
构建企业内部知识库，员工可以通过对话方式快速查询公司政策、流程文档等信息。

### 场景 4：教育培训
作为学习助手，回答学生问题。加载教材和课程资料到知识库，提供个性化辅导。

## 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone <repository>
cd LangChainProject

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入 OpenAI API Key
```

### 2. 启动服务
```bash
python app/main.py
```

服务将在 http://localhost:8000 启动。

### 3. 发送第一条消息
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好", "session_id": "test"}'
```

### 4. 启用 RAG 功能
```bash
# 1. 创建文档目录
mkdir -p data/documents

# 2. 放入文档
cp your-docs/* data/documents/

# 3. 修改 .env
RAG_ENABLED=true
RAG_REBUILD_ON_STARTUP=true

# 4. 重启服务
python app/main.py
```

## API 接口

### 对话接口
- `POST /chat` - 普通对话
- `POST /chat/stream` - 流式对话

### 历史管理
- `GET /history/{session_id}` - 获取历史
- `DELETE /history/{session_id}` - 清除历史

### RAG 接口
- `POST /rag/query` - 知识库问答
- `POST /rag/rebuild` - 重建知识库
- `GET /rag/stats` - 系统统计

### 系统管理
- `GET /health` - 健康检查
- `GET /tools` - 查看工具
- `GET /sessions` - 会话列表

详细 API 文档请访问：http://localhost:8000/docs

## 配置说明

### 必需配置
- `OPENAI_API_KEY` - OpenAI API 密钥
- `OPENAI_BASE_URL` - API 端点（可选）
- `MODEL_NAME` - 使用的模型名称

### 可选配置
- `ENABLE_TOOLS` - 是否启用工具调用
- `CONTEXT_COMPRESSION_STRATEGY` - 上下文压缩策略
- `RAG_ENABLED` - 是否启用 RAG
- `LANGFUSE_ENABLED` - 是否启用 Langfuse 监控

完整配置说明请查看 `.env.example` 文件。

## 常见问题

### Q: 如何选择合适的压缩策略？
A: 
- 简单问答：sliding_window（5-10轮）
- 技术支持：token_limit（4000-6000 tokens）
- 长期对话：summary（15-20轮触发）

### Q: RAG 支持哪些文档格式？
A: 目前支持 .txt、.md、.pdf 格式。更多格式支持正在开发中。

### Q: 如何降低 API 成本？
A: 
- 使用 gpt-4o-mini 等较便宜的模型
- 启用上下文压缩
- 设置合理的 RAG 检索数量（RAG_TOP_K）
- 降低 Langfuse 采样率

### Q: 服务可以部署到生产环境吗？
A: 可以，但建议：
- 使用 PostgreSQL 替代 SQLite
- 启用 HTTPS
- 配置反向代理（Nginx）
- 设置 RAG_REBUILD_ON_STARTUP=false
- 配置监控和日志

## 技术架构

### 核心组件
- **LangChain**: 提供 LLM 抽象和工具调用
- **LangGraph**: 实现对话流程编排
- **FastAPI**: Web 服务框架
- **SQLite/PostgreSQL**: 会话持久化
- **Chroma**: 向量数据库
- **Langfuse**: 监控和追踪

### 数据流
```
用户请求 → FastAPI → Agent → LangGraph → LLM
                ↓              ↓
           SQLite 存储    工具调用/RAG检索
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues
- Email: support@example.com

## 更新日志

### v1.0.0 (2024-01)
- 初始版本发布
- 支持基本对话功能
- 集成工具调用
- 添加 RAG 支持
