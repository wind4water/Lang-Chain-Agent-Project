# LangChain对话Agent项目

这是一个功能完整的LangChain对话Agent项目，基于LangGraph构建，支持工具调用、会话管理和持久化存储。

## 功能特性

- ✅ **LangChain Core**: 使用ChatOpenAI、Prompt模板、OutputParser等核心组件
- ✅ **LangGraph**: 基于状态图构建对话流程
- ✅ **Checkpoint**: 使用SqliteSaver实现会话持久化
- ✅ **HTTP API**: FastAPI提供RESTful接口
- ✅ **会话管理**: 支持多用户多会话，独立存储对话历史
- ✅ **异步处理**: 全异步架构，高性能
- ⭐ **Tool Calling**: Agent自动判断并调用工具完成任务（日期查询、时间戳等）
- ⭐ **上下文压缩**: 支持3种压缩策略（滑动窗口、Token计数、智能摘要），防止token超限

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
│   │   ├── builtin.py           # 内置工具定义（日期、时间戳等）
│   │   └── loader.py            # 工具加载器
│   └── main.py                  # FastAPI HTTP服务入口
├── tests/                       # 测试脚本
│   └── test_tool_calling.py     # ⭐ 工具调用测试
├── doc/                         # 文档目录
│   ├── STORAGE.md               # 存储位置详细说明
│   ├── STORAGE_OPTIONS.md       # 所有存储方案对比
│   ├── STORAGE_COMPARISON.md    # 快速对比和切换指南
│   ├── SWITCHED_TO_SQLITE.md    # SQLite切换指南
│   ├── POSTGRES_SETUP.md        # PostgreSQL部署指南
│   ├── DATABASE_QUERY_API.md    # 数据库查询接口文档
│   ├── COMPRESSION_USAGE.md     # 上下文压缩使用指南
│   ├── CONTEXT_COMPRESSION.md   # 压缩技术详解
│   ├── TOOL_CALLING.md          # ⭐ 工具调用完整文档
│   └── TOOL_CALLING_INTRO.md    # 工具调用快速入门
├── checkpoints/                 # Checkpoint数据库目录（自动创建）
├── requirements.txt             # 项目依赖
├── .env                         # 环境变量（需要自己创建）
├── .env.example                 # 环境变量示例
└── README.md                    # 项目文档（本文件）
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

# ========================================
# 上下文压缩策略（可选）
# ========================================
# 压缩策略选择: none | sliding_window | token_limit | summary
CONTEXT_COMPRESSION_STRATEGY=sliding_window  # 推荐使用滑动窗口
COMPRESSION_WINDOW_SIZE=10  # 滑动窗口保留的对话轮数
```

> 💡 **工具调用**：启用后Agent可以自动调用工具获取实时信息（如当前时间、日期等）。详见 [doc/TOOL_CALLING.md](doc/TOOL_CALLING.md)
> 
> 💡 **上下文压缩**：长对话会导致token超限和成本暴涨。推荐使用 `sliding_window` 策略。详见 [doc/COMPRESSION_USAGE.md](doc/COMPRESSION_USAGE.md)

### 3. 启动服务

```bash
python app/main.py
```

服务将在 `http://localhost:8000` 启动

启动日志示例：
```
🔧 加载了 3 个工具:
   - get_current_date_tool: 获取当前日期和时间
   - get_current_timestamp: 获取当前 Unix 时间戳
   - get_weekday: 获取今天是星期几
✅ 配置持久化存储: /Users/shenyang/Trip/AIProject/LangChainProject/checkpoints/conversations.db
✅ 上下文压缩策略: sliding_window
✅ 工具调用: 启用
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
```

响应：
```json
{
  "response": "现在是 2026年05月08日 14:30:45",
  "session_id": "user_123"
}
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
  "total": 3,
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
  "tools_count": 3
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

> 更多接口详情请查看：[doc/DATABASE_QUERY_API.md](doc/DATABASE_QUERY_API.md)

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

> 详细文档：[doc/COMPRESSION_USAGE.md](doc/COMPRESSION_USAGE.md)  
> 技术原理：[doc/CONTEXT_COMPRESSION.md](doc/CONTEXT_COMPRESSION.md)

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

1. ~~**添加工具调用**~~: ✅ 已实现！支持日期、时间戳等工具
2. **扩展工具系统**: 添加更多工具（计算器、搜索、文件操作等）
3. **流式响应**: 使用SSE实现实时输出
4. **多模型支持**: 添加其他LLM提供商
5. **RAG功能**: 集成向量数据库和检索器
6. **用户认证**: 添加JWT或其他认证机制
7. **MCP集成**: 集成飞书、GitLab、Notion等MCP工具

## 注意事项

- Checkpoint数据库文件存储在 `checkpoints/` 目录
- 不同session_id的对话完全隔离
- 当前使用SQLite持久化存储（重启后数据保留）
- 生产环境建议使用PostgreSQL或Redis作为checkpoint后端
- 工具调用功能可通过 `ENABLE_TOOLS` 环境变量开关
- Agent会自动判断是否需要调用工具，无需手动指定
- 记得保护好你的API密钥

## 📚 详细文档

### Tool Calling（工具调用）⭐
- [工具调用完整文档](doc/TOOL_CALLING.md) - **推荐阅读**，工具系统详解、使用示例、自定义工具
- [工具调用快速入门](doc/TOOL_CALLING_INTRO.md) - 快速上手工具调用

### 存储相关
- [存储位置说明](doc/STORAGE.md) - session_id上下文存储在哪里
- [存储方案对比](doc/STORAGE_OPTIONS.md) - 所有持久化存储方案详解
- [快速切换指南](doc/STORAGE_COMPARISON.md) - 三种存储方案对比
- [SQLite使用指南](doc/SWITCHED_TO_SQLITE.md) - 当前使用的SQLite配置
- [PostgreSQL部署](doc/POSTGRES_SETUP.md) - 生产环境PostgreSQL配置
- [数据库查询API](doc/DATABASE_QUERY_API.md) - 查询SQLite数据的接口文档

### 上下文压缩
- [压缩使用指南](doc/COMPRESSION_USAGE.md) - **推荐阅读**，配置和使用3种压缩策略
- [压缩技术详解](doc/CONTEXT_COMPRESSION.md) - 深入理解压缩原理和方案对比
- [Checkpoint机制](doc/CHECKPOINT_MECHANISM.md) - LangGraph自动保存机制详解

### 开发指南
- [踩坑记录](doc/PITFALLS.md) - SQLite持久化开发中的7个大坑
- [快速参考](doc/QUICK_REFERENCE.md) - 常见错误和解决方案

## 🧪 测试脚本

```bash
# ⭐ 工具调用测试（推荐优先尝试）
python tests/test_tool_calling.py

# 基础对话测试
python test_agent.py

# 持久化功能测试
python test_persistence.py

# 验证重启后数据保留
python test_persistence_verify.py

# 数据库查询接口测试
python test_query_database.py

# 会话隔离演示
python demo_session.py

# 存储位置检查
python check_storage.py

# 上下文压缩测试
python examples/test_compression.py --strategy all          # 对比所有策略
python examples/test_compression.py --strategy sliding_window  # 测试滑动窗口
python examples/test_compression.py --long                  # 测试超长对话（50轮）
```

## License

MIT
