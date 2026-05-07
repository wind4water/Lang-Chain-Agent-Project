# Session上下文存储说明

## 📦 当前实现：MemorySaver（内存存储）

### 存储位置
```
服务器内存（RAM）
├── Python进程
    └── FastAPI应用
        └── agent实例
            └── checkpointer (MemorySaver)
                └── 内部字典 {
                    "session_1": StateSnapshot,
                    "session_2": StateSnapshot,
                    ...
                }
```

### 特点
| 特性 | 说明 |
|------|------|
| 📍 存储位置 | Python进程的堆内存 |
| ⚡ 读写速度 | 极快（纳秒级） |
| 💾 数据持久化 | ❌ 服务重启后丢失 |
| 🔄 多实例共享 | ❌ 无法跨进程共享 |
| 📊 容量限制 | 受限于服务器内存 |
| 🎯 适用场景 | 开发、测试、演示 |

### 查看数据
```bash
# 运行检查脚本
python check_storage.py
```

## 🗄️ 升级方案：SqliteSaver（文件存储）

### 存储位置
```
磁盘文件系统
└── checkpoints/
    └── conversations.db  ← SQLite数据库文件
        └── 表: checkpoints
            ├── thread_id (索引)
            ├── checkpoint_id
            ├── checkpoint (JSON blob)
            └── metadata
```

### 特点
| 特性 | 说明 |
|------|------|
| 📍 存储位置 | 磁盘文件 `checkpoints/conversations.db` |
| ⚡ 读写速度 | 快（毫秒级） |
| 💾 数据持久化 | ✅ 服务重启后仍存在 |
| 🔄 多实例共享 | ⚠️ 同一台机器可共享 |
| 📊 容量限制 | 几乎无限（受限于磁盘） |
| 🎯 适用场景 | 生产环境（单机部署） |

### 使用方法
```python
# main.py 修改为：
from agent_persistent import ConversationAgentWithPersistence

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = ConversationAgentWithPersistence()  # 使用持久化版本
    yield

# 查看数据库文件
ls -lh checkpoints/conversations.db

# 使用SQLite命令行查看数据
sqlite3 checkpoints/conversations.db
> SELECT * FROM checkpoints;
```

## 🚀 生产级方案对比

### 1. PostgreSQL（推荐生产环境）
```python
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
```
- ✅ 分布式部署
- ✅ 多实例共享
- ✅ 高可用
- ✅ 数据备份

### 2. Redis
```python
from langgraph.checkpoint.redis import RedisSaver
checkpointer = RedisSaver.from_conn_info(...)
```
- ✅ 极高性能
- ✅ 分布式
- ⚠️ 需配置持久化

### 3. SQLite（当前可用）
```python
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver(conn)
```
- ✅ 无需额外服务
- ✅ 持久化
- ⚠️ 单机部署

## 🔄 如何切换存储方式

### 方案1：保持当前（内存存储）
不需要修改，适合快速开发测试

### 方案2：升级为SQLite（文件存储）
```bash
# 修改 main.py
# 将：from agent import ConversationAgent
# 改为：from agent_persistent import ConversationAgentWithPersistence

# 然后重启服务
python main.py
```

## 📊 性能对比

| 操作 | MemorySaver | SqliteSaver | PostgresSaver |
|------|-------------|-------------|---------------|
| 读取历史 | 0.001ms | 1-5ms | 5-20ms |
| 写入消息 | 0.001ms | 5-10ms | 10-30ms |
| 重启恢复 | ❌ | ✅ | ✅ |
| 并发支持 | 低 | 中 | 高 |
| 适合用户数 | <100 | <10,000 | 无限 |

## 💡 建议

1. **开发阶段**：使用 MemorySaver（当前方案）
2. **测试阶段**：使用 SqliteSaver（运行 `agent_persistent.py`）
3. **生产环境**：使用 PostgresSaver 或 Redis

## 🧪 实验：查看存储位置

```bash
# 1. 使用当前MemorySaver版本
python main.py

# 2. 创建一些对话
python check_storage.py

# 3. 重启服务
# Ctrl+C 停止，然后重新启动
python main.py

# 4. 再次查询历史
curl http://localhost:8000/history/session_1
# 返回：{"session_id":"session_1","history":[]}  ← 数据丢失了

# 5. 如果想测试持久化，修改main.py使用agent_persistent
# 重启后数据仍然存在
```

## ❓ FAQ

**Q: 为什么不一开始就用SQLite？**
A: 为了避免异步SQLite的兼容性问题，先用MemorySaver演示核心功能。

**Q: 如何查看内存中的数据？**
A: 运行 `python check_storage.py`

**Q: 想要持久化怎么办？**
A: 修改 `main.py` 导入 `ConversationAgentWithPersistence`

**Q: 生产环境推荐什么？**
A: PostgreSQL + 多实例部署 + 负载均衡
