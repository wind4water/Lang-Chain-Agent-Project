"""
查看checkpoint内部存储的数据
"""
import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 80)
print("演示：session_id上下文的存储位置")
print("=" * 80)

# 先创建几个会话
print("\n1️⃣ 创建测试数据...")
sessions = {
    "session_1": "我叫张三",
    "session_2": "我叫李四",
    "session_3": "我叫王五"
}

for sid, msg in sessions.items():
    requests.post(f"{BASE_URL}/chat", json={"message": msg, "session_id": sid})
    print(f"   ✓ {sid}: {msg}")

# 查看每个会话的历史
print("\n2️⃣ 查看各会话的存储数据...")
for sid in sessions.keys():
    response = requests.get(f"{BASE_URL}/history/{sid}")
    history = response.json()
    print(f"\n   [{sid}] 共 {len(history['history'])} 条消息:")
    for msg in history['history']:
        print(f"      - {msg['role']}: {msg['content']}")

print("\n" + "=" * 80)
print("📦 存储位置说明")
print("=" * 80)
print("""
当前使用 MemorySaver，数据存储在：

┌─────────────────────────────────────────┐
│   Python进程内存（RAM）                  │
│                                          │
│   ┌──────────────────────────────────┐  │
│   │  FastAPI应用                      │  │
│   │  ├── agent对象                    │  │
│   │  │   └── checkpointer (MemorySaver)│ │
│   │  │       └── storage (dict)       │  │
│   │  │           ├── "session_1": ... │  │
│   │  │           ├── "session_2": ... │  │
│   │  │           └── "session_3": ... │  │
│   └──────────────────────────────────┘  │
└─────────────────────────────────────────┘

特点：
✅ 优点：
   • 读写速度极快（内存操作）
   • 无需配置数据库
   • 适合开发和演示

❌ 缺点：
   • 服务重启后数据丢失
   • 无法跨多个服务实例共享
   • 内存有限（不适合大规模生产）

💡 生产环境建议使用：
   • SqliteSaver - 文件数据库（单机）
   • PostgresSaver - PostgreSQL数据库（分布式）
   • RedisSaver - Redis缓存（高性能）
""")

print("\n3️⃣ 测试：重启服务会丢失数据")
print("   请手动测试：")
print("   1. 按 Ctrl+C 停止服务")
print("   2. 重新运行 python main.py")
print("   3. 再次查询历史：curl http://localhost:8000/history/session_1")
print("   4. 你会发现历史记录消失了（返回空数组）")
