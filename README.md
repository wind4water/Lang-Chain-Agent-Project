# LangChain对话Agent项目

这是一个完整的LangChain对话Agent项目，使用LangGraph和Checkpoint实现会话管理。

## 功能特性

- ✅ **LangChain Core**: 使用ChatOpenAI、Prompt模板、OutputParser等核心组件
- ✅ **LangGraph**: 基于状态图构建对话流程
- ✅ **Checkpoint**: 使用SqliteSaver实现会话持久化
- ✅ **HTTP API**: FastAPI提供RESTful接口
- ✅ **会话管理**: 支持多用户多会话，独立存储对话历史
- ✅ **异步处理**: 全异步架构，高性能

## 项目结构

```
LangChainProject/
├── main.py              # FastAPI HTTP服务
├── agent.py             # LangChain Agent核心逻辑
├── requirements.txt     # 项目依赖
├── .env                 # 环境变量（需要自己创建）
├── .env.example         # 环境变量示例
├── checkpoints/         # Checkpoint数据库目录（自动创建）
└── README.md            # 项目文档
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
```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
```

### 3. 启动服务

```bash
python main.py
```

服务将在 `http://localhost:8000` 启动

### 4. 访问API文档

打开浏览器访问：`http://localhost:8000/docs`

## API接口

### 1. 对话接口

**POST** `/chat`

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你好，请介绍一下自己",
    "session_id": "user_123"
  }'
```

响应：
```json
{
  "response": "你好！我是一个AI助手...",
  "session_id": "user_123"
}
```

### 2. 获取历史记录

**GET** `/history/{session_id}`

```bash
curl "http://localhost:8000/history/user_123"
```

### 3. 清除历史记录

**DELETE** `/history/{session_id}`

```bash
curl -X DELETE "http://localhost:8000/history/user_123"
```

### 4. 健康检查

**GET** `/health`

```bash
curl "http://localhost:8000/health"
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

## 示例对话

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

## 扩展建议

1. **添加工具调用**: 集成LangChain Tools
2. **流式响应**: 使用SSE实现实时输出
3. **多模型支持**: 添加其他LLM提供商
4. **RAG功能**: 集成向量数据库和检索器
5. **用户认证**: 添加JWT或其他认证机制

## 注意事项

- Checkpoint数据库文件存储在 `checkpoints/` 目录
- 不同session_id的对话完全隔离
- 生产环境建议使用PostgreSQL或Redis作为checkpoint后端
- 记得保护好你的API密钥

## License

MIT
