# SQLite数据库查询接口文档

## 🆕 新增接口

### 1. 查询所有会话列表

**接口**: `GET /sessions`

**描述**: 返回数据库中存储的所有session_id

**请求示例**:
```bash
curl http://localhost:8000/sessions
```

**响应示例**:
```json
{
  "total": 3,
  "sessions": [
    "user_alice",
    "user_bob",
    "user_charlie"
  ]
}
```

---

### 2. 获取数据库统计信息 ⭐

**接口**: `GET /database/stats`

**描述**: 返回数据库的完整统计信息，包括：
- 总会话数
- 总checkpoint数
- 数据库文件大小
- 每个会话的详细统计

**请求示例**:
```bash
curl http://localhost:8000/database/stats
```

**响应示例**:
```json
{
  "total_sessions": 3,
  "total_checkpoints": 12,
  "database_size_bytes": 24576,
  "database_size_mb": 0.02,
  "database_path": "/path/to/checkpoints/conversations.db",
  "sessions": [
    {
      "session_id": "user_alice",
      "checkpoint_count": 5,
      "first_seen": "2024-05-07 10:30:00",
      "last_seen": "2024-05-07 11:45:00"
    },
    {
      "session_id": "user_bob",
      "checkpoint_count": 4,
      "first_seen": "2024-05-07 10:35:00",
      "last_seen": "2024-05-07 11:20:00"
    },
    {
      "session_id": "user_charlie",
      "checkpoint_count": 3,
      "first_seen": "2024-05-07 11:00:00",
      "last_seen": "2024-05-07 11:30:00"
    }
  ]
}
```

---

### 3. 查询指定会话的详细信息

**接口**: `GET /database/sessions/{session_id}`

**描述**: 返回指定会话的底层checkpoint信息

**请求示例**:
```bash
curl http://localhost:8000/database/sessions/user_alice
```

**响应示例**:
```json
{
  "session_id": "user_alice",
  "checkpoint_count": 5,
  "checkpoints": [
    {
      "checkpoint_id": "1ef0...",
      "parent_checkpoint_id": "1eef...",
      "timestamp": "2024-05-07 11:45:00",
      "has_data": true
    },
    {
      "checkpoint_id": "1eef...",
      "parent_checkpoint_id": "1eee...",
      "timestamp": "2024-05-07 11:40:00",
      "has_data": true
    }
  ]
}
```

---

## 📊 使用场景

### 场景1：系统监控
```bash
# 定期检查数据库状态
watch -n 60 'curl -s http://localhost:8000/database/stats | jq'
```

### 场景2：查找活跃用户
```bash
# 获取统计信息，按checkpoint数排序
curl -s http://localhost:8000/database/stats | jq '.sessions | sort_by(.checkpoint_count) | reverse'
```

### 场景3：调试特定会话
```bash
# 1. 先列出所有会话
curl http://localhost:8000/sessions

# 2. 查看特定会话的checkpoint信息
curl http://localhost:8000/database/sessions/user_alice

# 3. 查看对话历史
curl http://localhost:8000/history/user_alice
```

### 场景4：数据清理
```bash
# 1. 查看所有会话
curl http://localhost:8000/sessions

# 2. 删除不活跃的会话
curl -X DELETE http://localhost:8000/history/old_session_id
```

---

## 🔍 与SQL直接查询的对比

### 使用API（推荐）
```bash
curl http://localhost:8000/database/stats
```

**优点**:
- ✅ 简单易用
- ✅ 返回格式化JSON
- ✅ 自动处理错误
- ✅ 不需要直接访问数据库文件

### 使用SQL
```bash
sqlite3 checkpoints/conversations.db "SELECT * FROM checkpoints"
```

**优点**:
- ✅ 更灵活的查询
- ✅ 可以做复杂分析

**缺点**:
- ⚠️ 需要安装sqlite3
- ⚠️ 需要了解表结构
- ⚠️ 输出格式不友好

---

## 📝 完整API列表

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/` | 查看所有可用接口 |
| POST | `/chat` | 发送消息进行对话 |
| GET | `/history/{session_id}` | 获取会话对话历史 |
| DELETE | `/history/{session_id}` | 清除会话历史 |
| **GET** | **`/sessions`** | **列出所有会话ID** ⭐新 |
| **GET** | **`/database/stats`** | **数据库统计信息** ⭐新 |
| **GET** | **`/database/sessions/{session_id}`** | **会话详细信息** ⭐新 |
| GET | `/health` | 健康检查 |

---

## 🧪 测试脚本

运行完整测试：
```bash
python test_query_database.py
```

这个脚本会：
1. 创建测试数据（如果没有）
2. 查询所有会话
3. 显示数据库统计
4. 查看第一个会话的详细信息
5. 显示对话历史

---

## 💡 实用技巧

### 技巧1：格式化JSON输出
```bash
# 安装jq（如果没有）
brew install jq  # macOS
sudo apt install jq  # Ubuntu

# 使用jq格式化输出
curl -s http://localhost:8000/database/stats | jq '.'
```

### 技巧2：监控数据库增长
```bash
# 创建监控脚本
cat > monitor.sh << 'EOF'
#!/bin/bash
while true; do
  clear
  echo "=== 数据库统计 ($(date)) ==="
  curl -s http://localhost:8000/database/stats | jq '{
    total_sessions,
    total_checkpoints,
    database_size_mb
  }'
  sleep 10
done
EOF

chmod +x monitor.sh
./monitor.sh
```

### 技巧3：导出会话列表到文件
```bash
# 导出所有会话ID
curl -s http://localhost:8000/sessions | jq -r '.sessions[]' > sessions.txt

# 统计每个会话的消息数
for session in $(cat sessions.txt); do
  count=$(curl -s "http://localhost:8000/history/$session" | jq '.history | length')
  echo "$session: $count messages"
done
```

### 技巧4：查找最活跃的会话
```bash
curl -s http://localhost:8000/database/stats | \
  jq -r '.sessions | sort_by(.checkpoint_count) | reverse | .[0] | 
    "最活跃会话: \(.session_id) (共\(.checkpoint_count)个checkpoints)"'
```

---

## 📚 相关文档

- `SWITCHED_TO_SQLITE.md` - SQLite持久化配置说明
- `STORAGE_OPTIONS.md` - 所有存储方案对比
- `README.md` - 项目整体文档

---

## 🐛 故障排查

### 问题1：返回空数据
```bash
# 检查数据库文件是否存在
ls -lh checkpoints/conversations.db

# 如果不存在，创建一些测试数据
python test_query_database.py
```

### 问题2：接口报错
```bash
# 检查服务是否正常运行
curl http://localhost:8000/health

# 查看服务日志
# (在运行 python main.py 的终端查看)
```

### 问题3：数据库文件损坏
```bash
# 检查数据库完整性
sqlite3 checkpoints/conversations.db "PRAGMA integrity_check;"

# 如果损坏，从备份恢复
cp checkpoints/backup.db checkpoints/conversations.db
```

---

## 🎯 下一步

1. **运行测试**: `python test_query_database.py`
2. **查看API文档**: 访问 `http://localhost:8000/docs`
3. **监控数据库**: 使用 `/database/stats` 接口

需要更多功能或有问题随时问我！
