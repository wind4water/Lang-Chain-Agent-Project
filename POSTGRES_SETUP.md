# PostgreSQL存储部署指南

## 🚀 快速启动

### 方法1：使用Docker（推荐）

```bash
# 1. 启动PostgreSQL容器
docker run -d \
  --name langchain-postgres \
  -e POSTGRES_USER=langchain \
  -e POSTGRES_PASSWORD=langchain \
  -e POSTGRES_DB=langchain_db \
  -p 5432:5432 \
  postgres:16

# 2. 验证连接
docker exec -it langchain-postgres psql -U langchain -d langchain_db
# 输入: \l 查看数据库列表
# 输入: \q 退出

# 3. 添加环境变量到 .env
echo "POSTGRES_URL=postgresql://langchain:langchain@localhost:5432/langchain_db" >> .env

# 4. 安装依赖
pip install psycopg[binary] asyncpg

# 5. 修改main.py使用PostgreSQL版本
# 将: from agent import ConversationAgent
# 改为: from agent_postgres import ConversationAgentPostgres
# 将: agent = ConversationAgent()
# 改为: agent = ConversationAgentPostgres()

# 6. 启动服务
python main.py
```

### 方法2：使用本地PostgreSQL

```bash
# macOS
brew install postgresql@16
brew services start postgresql@16

# Ubuntu/Debian
sudo apt install postgresql-16
sudo systemctl start postgresql

# 创建数据库
createdb -U postgres langchain_db

# 创建用户
psql -U postgres
> CREATE USER langchain WITH PASSWORD 'langchain';
> GRANT ALL PRIVILEGES ON DATABASE langchain_db TO langchain;
> \q
```

## 📊 数据库表结构

PostgreSQL会自动创建以下表：

```sql
-- checkpoints表（存储会话状态）
CREATE TABLE checkpoints (
    thread_id TEXT,
    checkpoint_id TEXT,
    parent_id TEXT,
    checkpoint JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_id)
);

CREATE INDEX idx_thread_id ON checkpoints(thread_id);
CREATE INDEX idx_created_at ON checkpoints(created_at);
```

## 🔍 查询会话数据

```bash
# 连接数据库
docker exec -it langchain-postgres psql -U langchain -d langchain_db

# 或本地连接
psql -U langchain -d langchain_db
```

### SQL查询示例

```sql
-- 查看所有会话ID
SELECT DISTINCT thread_id FROM checkpoints;

-- 查看某个会话的所有checkpoint
SELECT 
    thread_id,
    checkpoint_id,
    created_at,
    jsonb_array_length(checkpoint->'channel_values'->'messages') as message_count
FROM checkpoints
WHERE thread_id = 'user_123'
ORDER BY created_at DESC;

-- 查看会话总数
SELECT COUNT(DISTINCT thread_id) as total_sessions FROM checkpoints;

-- 查看最活跃的会话
SELECT 
    thread_id,
    COUNT(*) as checkpoint_count,
    MAX(created_at) as last_active
FROM checkpoints
GROUP BY thread_id
ORDER BY checkpoint_count DESC
LIMIT 10;

-- 查看某个会话的消息内容
SELECT 
    thread_id,
    checkpoint->'channel_values'->'messages' as messages
FROM checkpoints
WHERE thread_id = 'user_123'
ORDER BY created_at DESC
LIMIT 1;

-- 清理30天前的数据
DELETE FROM checkpoints 
WHERE created_at < NOW() - INTERVAL '30 days';
```

## 🛠️ 环境变量配置

在 `.env` 文件中添加：

```bash
# PostgreSQL配置
POSTGRES_URL=postgresql://langchain:langchain@localhost:5432/langchain_db

# 或者分开配置
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=langchain
POSTGRES_PASSWORD=langchain
POSTGRES_DB=langchain_db
```

## 🔧 生产环境优化

### 1. 连接池配置
```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

checkpointer = AsyncPostgresSaver.from_conn_string(
    postgres_url,
    pool_size=20,          # 连接池大小
    max_overflow=10,       # 最大溢出连接
    pool_timeout=30,       # 连接超时
    pool_recycle=3600      # 连接回收时间
)
```

### 2. 索引优化
```sql
-- 添加复合索引
CREATE INDEX idx_thread_checkpoint ON checkpoints(thread_id, checkpoint_id);

-- 添加部分索引（只索引最近的数据）
CREATE INDEX idx_recent_checkpoints 
ON checkpoints(created_at) 
WHERE created_at > NOW() - INTERVAL '7 days';
```

### 3. 分区表（大规模数据）
```sql
-- 按月份分区
CREATE TABLE checkpoints_2024_01 PARTITION OF checkpoints
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### 4. 定期清理
```sql
-- 创建清理任务（保留30天数据）
CREATE OR REPLACE FUNCTION cleanup_old_checkpoints()
RETURNS void AS $$
BEGIN
    DELETE FROM checkpoints 
    WHERE created_at < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- 定期执行（使用pg_cron）
SELECT cron.schedule('cleanup-checkpoints', '0 2 * * *', 
    'SELECT cleanup_old_checkpoints()');
```

## 🏗️ 高可用架构

### 主从复制
```yaml
架构：
  Primary (主库):
    - 处理所有写操作
    - 实时复制到从库
  
  Replica1 (从库1):
    - 处理读操作
    - 负载均衡
  
  Replica2 (从库2):
    - 处理读操作
    - 故障转移备份

连接策略：
  写操作: primary.db.com:5432
  读操作: replica-lb.db.com:5432 (负载均衡)
```

### Docker Compose示例
```yaml
version: '3.8'
services:
  postgres-primary:
    image: postgres:16
    environment:
      POSTGRES_USER: langchain
      POSTGRES_PASSWORD: langchain
      POSTGRES_DB: langchain_db
    volumes:
      - pg-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    command: postgres -c wal_level=replica

  postgres-replica:
    image: postgres:16
    environment:
      POSTGRES_USER: langchain
      POSTGRES_PASSWORD: langchain
    depends_on:
      - postgres-primary

volumes:
  pg-data:
```

## 📈 监控指标

### 关键指标
```sql
-- 会话统计
SELECT 
    COUNT(DISTINCT thread_id) as total_sessions,
    COUNT(*) as total_checkpoints,
    pg_size_pretty(pg_total_relation_size('checkpoints')) as table_size
FROM checkpoints;

-- 每日活跃会话
SELECT 
    DATE(created_at) as date,
    COUNT(DISTINCT thread_id) as active_sessions
FROM checkpoints
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- 平均消息数
SELECT AVG(message_count) as avg_messages_per_session
FROM (
    SELECT 
        thread_id,
        MAX(jsonb_array_length(checkpoint->'channel_values'->'messages')) as message_count
    FROM checkpoints
    GROUP BY thread_id
) t;
```

## 🐛 故障排查

### 常见问题

1. **连接失败**
```bash
# 检查PostgreSQL是否运行
docker ps | grep postgres
# 或
pg_isready -h localhost -p 5432

# 检查防火墙
sudo ufw allow 5432
```

2. **权限问题**
```sql
-- 检查权限
\du langchain

-- 授予权限
GRANT ALL PRIVILEGES ON DATABASE langchain_db TO langchain;
GRANT ALL ON ALL TABLES IN SCHEMA public TO langchain;
```

3. **性能问题**
```sql
-- 查看慢查询
SELECT * FROM pg_stat_statements 
ORDER BY total_exec_time DESC 
LIMIT 10;

-- 分析表
ANALYZE checkpoints;

-- 重建索引
REINDEX TABLE checkpoints;
```

## 🔐 安全建议

1. **使用强密码**
2. **限制远程访问**（修改 pg_hba.conf）
3. **启用SSL连接**
4. **定期备份**
```bash
# 备份
pg_dump -U langchain langchain_db > backup.sql

# 恢复
psql -U langchain langchain_db < backup.sql
```

## 📦 依赖包

添加到 `requirements.txt`:
```txt
psycopg[binary]==3.1.18
asyncpg==0.29.0
```

## ✅ 部署检查清单

- [ ] PostgreSQL服务已启动
- [ ] 数据库和用户已创建
- [ ] 环境变量已配置
- [ ] 依赖包已安装
- [ ] 连接测试通过
- [ ] 表自动创建成功
- [ ] 备份策略已设置
- [ ] 监控已配置
