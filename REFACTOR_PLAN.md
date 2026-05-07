# 项目架构重构方案

## 当前问题

根目录有13个Python文件，结构混乱：
- 3个agent实现混在一起
- 7个测试文件散落在根目录
- 演示和工具脚本没有分类

## 新架构设计

```
LangChainProject/
├── app/                          # 应用核心代码
│   ├── __init__.py
│   ├── agents/                   # Agent实现
│   │   ├── __init__.py
│   │   ├── memory.py             # MemorySaver版本（原agent.py）
│   │   ├── sqlite.py             # SQLite版本（原agent_persistent.py）
│   │   └── postgres.py           # PostgreSQL版本（原agent_postgres.py）
│   └── main.py                   # FastAPI应用（原main.py）
│
├── tests/                        # 测试文件
│   ├── __init__.py
│   ├── test_agent.py             # 基础Agent测试
│   ├── test_persistence.py       # 持久化测试
│   ├── test_persistence_verify.py # 持久化验证
│   ├── test_async_sqlite.py      # AsyncSQLite测试
│   ├── test_fix_verification.py  # 修复验证
│   ├── test_query_database.py    # 数据库查询测试
│   └── test_quick_verify.py      # 快速验证
│
├── examples/                     # 示例和演示
│   ├── __init__.py
│   ├── demo_session.py           # 会话隔离演示
│   └── check_storage.py          # 存储检查工具
│
├── doc/                          # 文档（已有）
│   ├── README.md
│   ├── PITFALLS.md
│   └── ...
│
├── checkpoints/                  # 数据库文件（运行时生成）
├── .env                          # 环境变量
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md                     # 项目主文档
└── run.py                        # 启动脚本（新增）
```

## 优势

### 1. 清晰的模块划分
- **app/**: 核心应用代码
- **tests/**: 所有测试集中管理
- **examples/**: 示例和工具脚本

### 2. Agent实现分离
- `app/agents/memory.py` - 内存存储（开发）
- `app/agents/sqlite.py` - SQLite（生产单机）
- `app/agents/postgres.py` - PostgreSQL（生产分布式）

### 3. 导入更清晰
```python
# 之前
from agent_persistent import ConversationAgentWithPersistence

# 之后
from app.agents.sqlite import SqliteAgent
```

### 4. 易于扩展
- 新增agent实现：在 `app/agents/` 添加
- 新增测试：在 `tests/` 添加
- 新增示例：在 `examples/` 添加

## 迁移计划

### 第1步：创建新目录结构
```bash
mkdir -p app/agents tests examples
touch app/__init__.py app/agents/__init__.py tests/__init__.py examples/__init__.py
```

### 第2步：移动Agent文件
```bash
mv agent.py app/agents/memory.py
mv agent_persistent.py app/agents/sqlite.py
mv agent_postgres.py app/agents/postgres.py
```

### 第3步：移动主应用
```bash
mv main.py app/main.py
```

### 第4步：移动测试文件
```bash
mv test_*.py tests/
```

### 第5步：移动示例文件
```bash
mv demo_session.py examples/
mv check_storage.py examples/
```

### 第6步：更新导入和引用
- 修改 `app/main.py` 中的导入
- 修改测试文件中的导入
- 更新 `README.md` 中的路径

### 第7步：创建启动脚本
创建 `run.py` 简化启动

## 兼容性

为了保持向后兼容，可以在根目录创建代理文件：
```python
# main.py（根目录）
from app.main import app
# 兼容旧的导入方式
```

## 替代方案

如果觉得 `app/` 太泛化，也可以用：
- `langchain_agent/` - 更具体的命名
- `src/` - 传统Python项目结构
- `chatbot/` - 业务相关命名

## 建议

✅ 推荐使用 `app/` 结构：
- 简洁清晰
- 符合FastAPI惯例
- 易于理解和维护
