"""
验证AsyncSqliteSaver修复
"""
import requests
import time
import json

BASE_URL = "http://localhost:8000"

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def test_api(endpoint, method="GET", data=None):
    """测试API"""
    try:
        if method == "POST":
            response = requests.post(f"{BASE_URL}{endpoint}", json=data)
        else:
            response = requests.get(f"{BASE_URL}{endpoint}")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

print_section("🔧 验证AsyncSqliteSaver修复")

# 1. 健康检查
print("\n1️⃣  健康检查...")
health = test_api("/health")
print(f"   状态: {health.get('status', 'unknown')}")
print(f"   存储类型: {health.get('storage_type', 'unknown')}")

# 2. 检查初始数据库状态
print("\n2️⃣  检查数据库初始状态...")
stats = test_api("/database/stats")
if "error" in stats:
    print(f"   ❌ 错误: {stats['error']}")
    print("\n   可能的问题：")
    print("   1. 服务未启动")
    print("   2. 数据库表未创建")
    print("\n   解决方法：重启服务 python main.py")
    exit(1)
else:
    print(f"   ✅ 数据库正常")
    print(f"   • 会话数: {stats['total_sessions']}")
    print(f"   • Checkpoint数: {stats['total_checkpoints']}")
    print(f"   • 数据库大小: {stats['database_size_mb']} MB")

# 3. 发送第一条消息
print("\n3️⃣  发送测试消息...")
result = test_api("/chat", "POST", {
    "message": "你好，我叫测试用户，今年25岁",
    "session_id": "test_async_001"
})

if "error" in result or "detail" in result:
    print(f"   ❌ 发送失败: {result}")
    exit(1)
else:
    print(f"   ✅ 消息已发送")
    print(f"   AI回复: {result.get('response', '')[:80]}...")

# 4. 等待写入
time.sleep(0.5)

# 5. 验证数据是否存储
print("\n4️⃣  验证数据持久化...")
stats = test_api("/database/stats")
print(f"   • 会话数: {stats['total_sessions']}")
print(f"   • Checkpoint数: {stats['total_checkpoints']}")

if stats['total_sessions'] > 0:
    print(f"   ✅ 数据已成功持久化！")
else:
    print(f"   ❌ 数据未持久化")

# 6. 查看会话列表
print("\n5️⃣  查看会话列表...")
sessions = test_api("/sessions")
print(f"   总会话: {sessions.get('total', 0)}")
if sessions.get('sessions'):
    for s in sessions['sessions']:
        print(f"      • {s}")

# 7. 测试多轮对话
print("\n6️⃣  测试多轮对话...")
test_api("/chat", "POST", {
    "message": "我刚才告诉你我叫什么名字？",
    "session_id": "test_async_001"
})
test_api("/chat", "POST", {
    "message": "我多大了？",
    "session_id": "test_async_001"
})

# 8. 查看对话历史
print("\n7️⃣  查看对话历史...")
history = test_api("/history/test_async_001")
if history.get('history'):
    print(f"   共 {len(history['history'])} 条消息:")
    for i, msg in enumerate(history['history'], 1):
        role = msg['role']
        content = msg['content'][:60] + "..." if len(msg['content']) > 60 else msg['content']
        print(f"      {i}. [{role}]: {content}")

# 9. 最终统计
print("\n8️⃣  最终统计...")
stats = test_api("/database/stats")
print(f"   • 总会话数: {stats['total_sessions']}")
print(f"   • 总Checkpoint数: {stats['total_checkpoints']}")
print(f"   • 数据库大小: {stats['database_size_mb']} MB")

if stats.get('sessions'):
    print(f"\n   会话详情:")
    for s in stats['sessions']:
        print(f"      • {s['session_id']}: {s['checkpoint_count']} checkpoints")

# 10. 结果
print_section("结果")
if stats['total_checkpoints'] > 0:
    print("✅ AsyncSqliteSaver工作正常！")
    print("✅ 数据已成功持久化到SQLite数据库")
    print(f"\n数据库位置: {stats['database_path']}")
    print("\n现在可以：")
    print("1. 重启服务测试数据是否保留")
    print("2. 使用 sqlite3 命令查看数据库内容")
    print("3. 继续添加更多会话进行测试")
else:
    print("❌ 持久化仍有问题")
    print("\n请检查：")
    print("1. 服务是否正常运行")
    print("2. 查看服务日志是否有错误")
    print("3. 确认 aiosqlite 已安装")
