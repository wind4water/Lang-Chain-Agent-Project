"""
测试SQLite持久化功能
验证：即使重启服务，对话历史仍然保留
"""
import requests
import time
import os

BASE_URL = "http://localhost:8000"

def chat(message: str, session_id: str = "test_persist"):
    """发送消息"""
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"message": message, "session_id": session_id}
    )
    result = response.json()
    print(f"用户: {message}")
    print(f"AI: {result['response']}\n")
    return result

def get_history(session_id: str = "test_persist"):
    """获取历史"""
    response = requests.get(f"{BASE_URL}/history/{session_id}")
    return response.json()

print("=" * 80)
print("🗄️  SQLite持久化测试")
print("=" * 80)

print("\n📝 第一步：创建对话历史")
print("-" * 80)
chat("你好，我叫张三，今年25岁，是一名程序员", session_id="user_persist")
chat("我喜欢Python和AI技术", session_id="user_persist")
chat("请记住我的信息", session_id="user_persist")

print("\n📊 第二步：查看当前历史记录")
print("-" * 80)
history = get_history("user_persist")
print(f"当前共有 {len(history['history'])} 条消息")
for i, msg in enumerate(history['history'], 1):
    print(f"  {i}. [{msg['role']}]: {msg['content'][:60]}...")

print("\n💾 第三步：检查SQLite数据库文件")
print("-" * 80)
db_path = "checkpoints/conversations.db"
if os.path.exists(db_path):
    size = os.path.getsize(db_path)
    print(f"✅ 数据库文件已创建: {db_path}")
    print(f"✅ 文件大小: {size} 字节")
    print(f"✅ 绝对路径: {os.path.abspath(db_path)}")
else:
    print(f"❌ 数据库文件不存在: {db_path}")

print("\n🔄 第四步：重启服务测试")
print("-" * 80)
print("请按照以下步骤操作：")
print("")
print("1️⃣  按 Ctrl+C 停止当前运行的服务")
print("2️⃣  重新运行: python main.py")
print("3️⃣  然后运行: python test_persistence_verify.py")
print("")
print("如果对话历史仍然存在，说明持久化成功！")

print("\n" + "=" * 80)
print("💡 提示")
print("=" * 80)
print("""
使用SQLite持久化后：
✅ 数据保存在磁盘文件中
✅ 服务重启后数据不丢失
✅ 可以直接备份.db文件
✅ 可以用SQLite工具查看数据

查看数据库内容：
  sqlite3 checkpoints/conversations.db
  > SELECT * FROM checkpoints;
  > .quit

备份数据：
  cp checkpoints/conversations.db checkpoints/backup_$(date +%Y%m%d).db
""")
