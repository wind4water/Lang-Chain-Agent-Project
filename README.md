# LangChain对话Agent项目

这是一个功能完整的LangChain对话Agent项目，基于LangGraph构建，支持工具调用、会话管理和持久化存储。

## 功能特性

- ✅ **LangChain Core**: 使用ChatOpenAI、Prompt模板、OutputParser等核心组件
- ✅ **LangGraph**: 基于状态图构建对话流程
- ✅ **Checkpoint**: 使用SqliteSaver实现会话持久化
- ✅ **HTTP API**: FastAPI提供RESTful接口
- ✅ **会话管理**: 支持多用户多会话，独立存储对话历史
- ✅ **异步处理**: 全异步架构，高性能
- ⭐ **Tool Calling**: Agent自动判断并调用工具完成任务（日期、搜索、天气、计算器、知识库等10个内置工具）
- ⭐ **RAG知识库**: 支持文档检索增强生成，Agent可自动查询内部知识库（TXT/MD/PDF格式）
- ⭐ **上下文压缩**: 支持3种压缩策略（滑动窗口、Token计数、智能摘要），防止token超限
- ⭐ **流式响应**: 支持SSE流式输出，实时显示AI回复
- ⭐ **Token统计**: 完整的Token使用统计和成本追踪（按会话、按天、按月）
- ⭐ **Langfuse监控**: 可选的云端监控和追踪（支持完整的trace追踪）
- ⭐ **Skill机制**: 支持按渠道加载策略（已内置飞书 RAG 优先 skill）

## 项目结构

```
LangChainProject/
├── app/                         # 应用代码目录
│   ├── agents/                  # Agent实现
│   │   ├── memory.py            # MemorySaver版本（内存）
│   │   ├── sqlite.py            # SQLite持久化版本（无工具）
│   │   ├── postgres.py          # PostgreSQL版本（生产环境）
│   │   └── sqlite_with_tools.py # ⭐ SQLite + Tool Calling（当前使用）
│   ├── tools/                   # ⭐ 工具系统
│   │   ├── __init__.py          # 工具导出
│   │   ├── builtin.py           # 内置工具定义
│   │   └── loader.py            # 工具加载器
│   ├── skills/                  # ⭐ Skill 系统（渠道策略）
│   │   ├── defs/                # 通用 skill（可提交）
│   │   ├── sensitive/           # 敏感 skill（默认忽略）
│   │   ├── loader.py            # skill 加载器
│   │   └── README.md            # skill 目录规范
│   └── main.py                  # FastAPI HTTP服务入口
│
├── tests/                       # 测试目录
│   ├── integration/             # 集成测试
│   │   ├── test_tool_calling.py # 工具调用测试
│   │   ├── test_persistence.py  # 持久化测试
│   │   └── ...
│   ├── unit/                    # 单元测试
│   │   ├── test_agent.py        # Agent单元测试
│   │   └── ...
│   └── demos/                   # 演示脚本
│       ├── demo_session.py      # 会话隔离演示
│       ├── test_compression.py  # 压缩策略演示
│       └── ...
│
├── docs/                        # 文档目录
│   ├── guides/                  # 使用指南
│   │   ├── tool-calling.md      # 工具调用完整文档
│   │   ├── compression.md       # 压缩使用指南
│   │   ├── storage.md           # 存储说明
│   │   └── ...
│   ├── api/                     # API文档
│   │   └── database-query-api.md
│   ├── troubleshooting/         # 故障排查
│   │   ├── langfuse-issues.md   # Langfuse问题
│   │   ├── pitfalls.md          # 开发踩坑
│   │   └── ...
│   └── architecture/            # 架构设计
│       ├── checkpoint-mechanism.md
│       └── ...
│
├── scripts/                     # 工具脚本
│   ├── dev/                     # 开发工具
│   │   ├── run.py
│   │   └── verify_compression.sh
│   └── test/                    # 测试脚本
│       ├── test_langfuse_fixed.py
│       └── ...
│
├── data/                        # 数据目录（自动创建）
│   ├── documents/               # RAG 文档目录
│   ├── chroma/                  # 向量数据库
│   └── checkpoints.db           # SQLite 数据库
├── requirements.txt             # 项目依赖
├── .env.example                 # 环境变量示例
└── README.md                    # 项目文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的API密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件：
```bash
# ========================================
# OpenAI API配置
# ========================================
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini

# ========================================
# 工具系统配置（⭐ 新功能）
# ========================================
# 是否启用工具调用功能
# true: Agent 可以自动调用工具（推荐）
# false: 纯对话模式，不调用工具
ENABLE_TOOLS=true
# 工具调用阶段模型温度（推荐低温，减少工具参数漂移）
TOOL_CALL_TEMPERATURE=0.1

# ========================================
# 上下文压缩策略（可选）
# ========================================
# 压缩策略选择: none | sliding_window | token_limit | summary
CONTEXT_COMPRESSION_STRATEGY=sliding_window  # 推荐使用滑动窗口
COMPRESSION_WINDOW_SIZE=10  # 滑动窗口保留的对话轮数

# ========================================
# RAG 知识库配置（可选）⭐
# ========================================
# 是否启用 RAG 知识库功能
RAG_ENABLED=true  # 默认启用，Agent可自动查询内部知识库
# 文档目录路径（支持 TXT、MD、PDF 格式）
RAG_DOCUMENTS_PATH=data/documents
# 向量数据库路径
RAG_CHROMA_PATH=data/chroma
# 是否在启动时重建索引
RAG_REBUILD_ON_STARTUP=false  # 仅首次启动或文档变更时设为 true

# ========================================
# 飞书 Skill 配置（可选）⭐
# ========================================
# 飞书渠道启用的 skill ID（对应 app/skills/defs/<id>.json）
FEISHU_SKILL_ID=feishu_rag_first

# ========================================
# Langfuse 监控配置（可选）
# ========================================
# 是否启用 Langfuse 追踪和监控
LANGFUSE_ENABLED=false  # 默认禁用，不影响现有功能
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key-here
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_SAMPLE_RATE=1.0  # 采样率 0.0-1.0
```

> 💡 **工具调用**：启用后Agent可以自动调用10个内置工具（日期、搜索、天气、计算器、知识库等）。详见 [docs/guides/tool-calling.md](docs/guides/tool-calling.md)
> 
> 💡 **RAG 知识库**：启用后Agent可自动查询内部文档（TXT/MD/PDF）。将文档放入 `data/documents/` 即可。详见 [docs/guides/rag-tool-integration.md](docs/guides/rag-tool-integration.md)
> 
> 💡 **上下文压缩**：长对话会导致token超限和成本暴涨。推荐使用 `sliding_window` 策略。详见 [docs/guides/compression.md](docs/guides/compression.md)
>
> 💡 **Langfuse 监控**：可选的云端监控和完整trace追踪。需要注册 [Langfuse](https://cloud.langfuse.com) 获取密钥。详见 [docs/troubleshooting/langfuse-issues.md](docs/troubleshooting/langfuse-issues.md)

## Skill 功能与配置

### 能力说明

- 支持按渠道定义对话策略（当前主要用于飞书）
- 当前默认 skill：`feishu_rag_first`
  - 先做 RAG 检索
  - 再按场景决定是否补充工具
  - 最后模型总结答案
  - 对外回复仅输出答案，不输出来源

### 目录与文件约定

- 通用 skill（可提交）：`app/skills/defs/*.json`
- 敏感 skill（自动忽略）：
  - `app/skills/sensitive/`
  - `app/skills/**/*.sensitive.json`
- 模板文件：`app/skills/defs/skill_template.json`

### 关键配置项

- `FEISHU_SKILL_ID`
  - 说明：飞书渠道当前生效的 skill
  - 示例：`FEISHU_SKILL_ID=feishu_rag_first`
- `TOOL_CALL_TEMPERATURE`
  - 说明：工具调用阶段模型温度，建议低值减少参数改写漂移
  - 推荐：`0.0 ~ 0.2`（默认 `0.1`）

### 操作流程

1. 在 `app/skills/defs/` 新增或复制一个 skill JSON（可参考 `skill_template.json`）
2. 在 `.env` 设置 `FEISHU_SKILL_ID=<你的skill_id>`
3. 重启服务使配置生效
4. 在飞书发起问题验证策略是否符合预期
5. 如需重置当前群上下文，在飞书发送 `/clear` 或 `/reset`

### 常见场景建议

- 内部知识问答：优先 RAG，禁用或弱化外部搜索
- 实时信息问题（天气/汇率/新闻）：允许工具补充
- 高稳定性场景：降低 `TOOL_CALL_TEMPERATURE`，减少工具参数漂移

### 上线前检查清单

- 确认 `FEISHU_SKILL_ID` 指向正确 skill，且对应 JSON 文件存在
- 确认 `RAG_ENABLED=true` 且知识库已完成同步（必要时执行一次 `POST /rag/sync`）
- 抽样验证 3 类问题：内部知识、实时信息、普通闲聊
- 在飞书群测试 `/clear` 或 `/reset`，确认上下文可被重置
- 检查 Langfuse（如启用）是否能看到完整链路与关键元数据

### 3. 启动服务

```bash
python app/main.py
```

服务将在 `http://localhost:8000` 启动

启动日志示例：
```
🔧 加载了 10 个工具:
   - get_current_date_tool: 获取当前日期和时间
   - get_current_timestamp: 获取当前 Unix 时间戳
   - get_weekday: 获取今天是星期几
   - calculator: 安全的数学计算器，支持基本运算
   - web_search: 使用 DuckDuckGo 进行网络搜索
   - get_weather: 获取指定城市的天气信息
   - currency_converter: 货币汇率转换
   - fetch_url_content: 抓取网页内容
   - python_executor: 安全的 Python 代码执行器
   - knowledge_base_search: 在内部知识库中搜索信息  ⭐ RAG
✅ 配置持久化存储: /Users/shenyang/Trip/AIProject/LangChainProject/checkpoints/conversations.db
✅ 上下文压缩策略: sliding_window
✅ 工具调用: 启用
✅ Langfuse 监控已启用 (采样率: 100%)  # 如果启用了 Langfuse
✅ LangChain Agent initialized with Tool Calling + SQLite checkpoint support
```

### 4. 访问API文档

打开浏览器访问：`http://localhost:8000/docs`

## API接口

### 核心功能接口

### 1. 对话接口（支持工具调用）

**POST** `/chat`

Agent会自动判断是否需要调用工具来完成任务。

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你好，请介绍一下自己",
    "session_id": "user_123"
  }'
```

工具调用示例：
```bash
# 查询当前时间（Agent会自动调用get_current_date_tool）
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "现在几点了？",
    "session_id": "user_123"
  }'

# 搜索信息（Agent会自动调用web_search）
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "帮我搜索 Python 最新版本",
    "session_id": "user_123"
  }'

# 查询天气（Agent会自动调用get_weather）
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "北京今天天气怎么样？",
    "session_id": "user_123"
  }'

# 查询知识库（Agent会自动调用knowledge_base_search）⭐
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "系统支持哪些文档格式？",
    "session_id": "user_123"
  }'
```

响应：
```json
{
  "response": "现在是 2026年05月08日 14:30:45",
  "session_id": "user_123"
}
```

**POST** `/chat/stream` - 流式响应（SSE）⭐

实时逐字输出AI回复，提供更好的用户体验。

```bash
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "讲一个故事",
    "session_id": "user_123"
  }'
```

响应格式（Server-Sent Events）：
```
data: {"content": "从"}
data: {"content": "前"}
data: {"content": "有"}
...
data: [DONE]
```

### 2. 查看可用工具 ⭐

**GET** `/tools`

```bash
curl "http://localhost:8000/tools"
```

响应：
```json
{
  "enabled": true,
  "total": 10,
  "tools": [
    {
      "name": "get_current_date_tool",
      "description": "获取当前日期和时间"
    },
    {
      "name": "get_current_timestamp",
      "description": "获取当前 Unix 时间戳"
    },
    {
      "name": "get_weekday",
      "description": "获取今天是星期几"
    },
    {
      "name": "calculator",
      "description": "安全的数学计算器，支持基本运算"
    },
    {
      "name": "web_search",
      "description": "使用 DuckDuckGo 进行网络搜索"
    },
    {
      "name": "get_weather",
      "description": "获取指定城市的天气信息"
    },
    {
      "name": "currency_converter",
      "description": "货币汇率转换"
    },
    {
      "name": "fetch_url_content",
      "description": "抓取网页内容"
    },
    {
      "name": "python_executor",
      "description": "安全的 Python 代码执行器"
    },
    {
      "name": "knowledge_base_search",
      "description": "在内部知识库中搜索信息"
    }
  ]
}
```

### 3. 获取历史记录

**GET** `/history/{session_id}`

```bash
curl "http://localhost:8000/history/user_123"
```

### 4. 清除历史记录

**DELETE** `/history/{session_id}`

```bash
curl -X DELETE "http://localhost:8000/history/user_123"
```

### 数据库查询接口

### 5. 健康检查（包含工具状态）

**GET** `/health`

```bash
curl "http://localhost:8000/health"
```

响应：
```json
{
  "status": "healthy",
  "agent_initialized": true,
  "storage_type": "SQLite",
  "tools_enabled": true,
  "tools_count": 10,
  "langfuse_enabled": false,
  "langfuse_sample_rate": 0.0
}
```

### 6. 查询所有会话

**GET** `/sessions`

```bash
curl "http://localhost:8000/sessions"
```

### 7. 数据库统计信息

**GET** `/database/stats`

```bash
curl "http://localhost:8000/database/stats"
```

> 更多接口详情请查看：[docs/api/database-query-api.md](docs/api/database-query-api.md)

### Token 使用统计 ⭐

### 8. 查询会话Token统计

**GET** `/stats/tokens/{session_id}`

查看指定会话的Token消耗和成本。

```bash
curl "http://localhost:8000/stats/tokens/user_123"
```

响应示例：
```json
{
  "session_id": "user_123",
  "request_count": 15,
  "total_prompt_tokens": 3250,
  "total_completion_tokens": 1820,
  "total_tokens": 5070,
  "total_cost_usd": 0.001521,
  "by_model": [
    {
      "model_name": "gpt-4o-mini",
      "request_count": 15,
      "prompt_tokens": 3250,
      "completion_tokens": 1820,
      "total_tokens": 5070,
      "cost_usd": 0.001521
    }
  ]
}
```

### 9. 每日Token统计

**GET** `/stats/tokens/daily?date=2026-05-09`

```bash
curl "http://localhost:8000/stats/tokens/daily?date=2026-05-09"
```

### 10. 每月Token统计

**GET** `/stats/tokens/monthly?year_month=2026-05`

```bash
curl "http://localhost:8000/stats/tokens/monthly?year_month=2026-05"
```

## 核心技术说明

### 1. LangChain组件使用

- **ChatOpenAI**: LLM模型
- **ChatPromptTemplate**: Prompt模板管理
- **MessagesPlaceholder**: 动态消息占位符
- **StrOutputParser**: 输出解析器

### 2. LangGraph状态图

```python
StateGraph(State)  # 定义状态图
  ├── chat节点     # 处理对话逻辑
  └── checkpoint   # 自动保存状态
```

### 3. Checkpoint机制

- 使用 **SqliteSaver** 持久化会话状态
- 每个session_id独立存储
- 自动管理对话历史
- 支持跨请求恢复上下文

### 4. 状态管理

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

使用 `add_messages` 自动累积消息历史

### 5. 上下文压缩策略 ⭐

**问题**：长对话会导致token超限、成本暴涨、响应变慢

**解决方案**：3种压缩策略可选

| 策略 | 适用场景 | Token消耗 | 信息保留 |
|------|---------|-----------|---------|
| **none** | 短对话、测试 | 无限增长 | 100% |
| **sliding_window** | 客服、问答（推荐）| 固定 | 70% |
| **token_limit** | 代码助手 | 精确控制 | 75% |
| **summary** | 项目管理、咨询 | 大幅减少 | 90% |

**快速开始**：

```bash
# .env 中配置
CONTEXT_COMPRESSION_STRATEGY=sliding_window
COMPRESSION_WINDOW_SIZE=10  # 保留最近10轮对话
```

启动后会看到压缩效果：
```
🔄 [滑动窗口] 压缩: 30 → 20 条消息
```

> 详细文档：[docs/guides/compression.md](docs/guides/compression.md)  
> 技术原理：[docs/guides/context-compression.md](docs/guides/context-compression.md)

## 示例对话

### 基础对话（会话记忆）

```python
# 第一轮对话
POST /chat
{
  "message": "我叫张三",
  "session_id": "user_123"
}

# 第二轮对话（Agent会记住你的名字）
POST /chat
{
  "message": "我叫什么名字？",
  "session_id": "user_123"
}
# 回复：根据前面的对话，你叫张三。
```

### 工具调用示例 ⭐

```python
# 查询时间（自动调用 get_current_date_tool）
POST /chat
{
  "message": "现在几点了？",
  "session_id": "user_123"
}
# 回复：现在是 2026年05月08日 14:30:45

# 查询星期（自动调用 get_weekday）
POST /chat
{
  "message": "今天星期几？",
  "session_id": "user_123"
}
# 回复：今天是星期四

# 不需要工具的对话
POST /chat
{
  "message": "什么是人工智能？",
  "session_id": "user_123"
}
# 回复：人工智能是...（直接回答，不调用工具）
```

## 扩展建议

1. ~~**添加工具调用**~~: ✅ 已实现！支持10个内置工具（日期、搜索、天气、计算器、知识库等）
2. ~~**流式响应**~~: ✅ 已实现！使用SSE实现实时输出（`/chat/stream`）
3. ~~**Token统计**~~: ✅ 已实现！完整的Token使用统计和成本追踪
4. ~~**监控追踪**~~: ✅ 已实现！可选的Langfuse云端监控
5. ~~**RAG功能**~~: ✅ 已实现！集成向量数据库和文档检索（支持TXT/MD/PDF）
6. **扩展工具系统**: 添加更多工具（文件操作、数据库查询等）
7. **多模型支持**: 添加其他LLM提供商（Anthropic Claude、Gemini等）
8. **用户认证**: 添加JWT或其他认证机制
9. **MCP集成**: 集成飞书、GitLab、Notion等MCP工具

## 注意事项

- Checkpoint数据库文件存储在 `checkpoints/` 目录
- 不同session_id的对话完全隔离
- 当前使用SQLite持久化存储（重启后数据保留）
- 生产环境建议使用PostgreSQL或Redis作为checkpoint后端
- 工具调用功能可通过 `ENABLE_TOOLS` 环境变量开关
- Agent会自动判断是否需要调用工具，无需手动指定
- Token统计数据保存在SQLite数据库中，支持按会话、按天、按月查询
- Langfuse监控是可选功能，配置错误或连接失败不会影响对话功能
- ⚠️ Langfuse与LangGraph存在兼容性问题，可能输出一些内部错误信息（不影响功能），详见 [docs/troubleshooting/langfuse-issues.md](docs/troubleshooting/langfuse-issues.md)
- 记得保护好你的API密钥

## 📚 详细文档

### RAG 知识库 ⭐
- [RAG 工具集成文档](docs/guides/rag-tool-integration.md) - RAG功能使用指南和示例
- [Agent 决策机制](docs/guides/agent-decision-mechanism.md) - Agent如何判断调用RAG
- [RAG 架构分析](docs/architecture/rag-implementation-analysis.md) - 当前实现和改进方向
- [企业级 RAG 架构](docs/architecture/enterprise-rag-architecture.md) - 大厂RAG系统参考

### Tool Calling（工具调用）⭐
- [工具调用完整文档](docs/guides/tool-calling.md) - **推荐阅读**，工具系统详解、使用示例、自定义工具
- [工具调用快速入门](docs/guides/tool-calling-intro.md) - 快速上手工具调用

### 监控与统计 ⭐
- [Langfuse问题排查](docs/troubleshooting/langfuse-issues.md) - Langfuse集成的已知问题和解决方案
- [Langfuse修复总结](docs/troubleshooting/langfuse-fix-summary.md) - 错误修复的技术细节
- Token统计 - 完整的Token使用统计API（见上方API接口文档）

### 存储相关
- [存储位置说明](docs/guides/storage.md) - session_id上下文存储在哪里
- [存储方案对比](docs/guides/storage-options.md) - 所有持久化存储方案详解
- [快速切换指南](docs/guides/storage-comparison.md) - 三种存储方案对比
- [SQLite使用指南](docs/guides/switched-to-sqlite.md) - 当前使用的SQLite配置
- [PostgreSQL部署](docs/guides/postgres-setup.md) - 生产环境PostgreSQL配置
- [数据库查询API](docs/api/database-query-api.md) - 查询SQLite数据的接口文档

### 上下文压缩
- [压缩使用指南](docs/guides/compression.md) - **推荐阅读**，配置和使用3种压缩策略
- [压缩技术详解](docs/guides/context-compression.md) - 深入理解压缩原理和方案对比
- [Checkpoint机制](docs/architecture/checkpoint-mechanism.md) - LangGraph自动保存机制详解

### 开发指南
- [踩坑记录](docs/troubleshooting/pitfalls.md) - SQLite持久化开发中的7个大坑
- [快速参考](docs/troubleshooting/quick-reference.md) - 常见错误和解决方案

## 🧪 测试脚本

```bash
# ⭐ 工具调用测试（推荐优先尝试）
python tests/integration/test_tool_calling.py

# ⭐ RAG 工具测试
python tests/test_rag_tool_quick.py

# ⭐ RAG 集成演示（需先启动服务）
bash scripts/demo_rag_tool.sh

# ⭐ Langfuse修复验证
python scripts/test/test_langfuse_fixed.py

# 基础对话测试
python tests/unit/test_agent.py

# 持久化功能测试
python tests/integration/test_persistence.py

# 验证重启后数据保留
python tests/integration/test_persistence_verify.py

# 数据库查询接口测试
python tests/integration/test_query_database.py

# 会话隔离演示
python tests/demos/demo_session.py

# 存储位置检查
python tests/demos/check_storage.py

# 上下文压缩测试
python tests/demos/test_compression.py --strategy all          # 对比所有策略
python tests/demos/test_compression.py --strategy sliding_window  # 测试滑动窗口
python tests/demos/test_compression.py --long                  # 测试超长对话（50轮）
```

## License

MIT
