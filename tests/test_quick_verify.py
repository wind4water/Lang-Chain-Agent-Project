"""
快速验证AsyncSqliteSaver
"""
import requests
import time

BASE_URL = "http://localhost:8000"

print("=" * 60)
print("🔧 快速验证 AsyncSqliteSaver")
print("=" * 60)

try:
    # 1. 发送消息
    print("\n1️⃣  发送测试消息...")
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"message": "你好，我是测试用户", "session_id": "quick_test"}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ 成功！")
        print(f"   回复: {result['response'][:50]}...")
    else:
        print(f"   ❌ 失败: {response.text}")
        exit(1)

    # 2. 等待写入
    time.sleep(0.5)

    # 3. 查看统计
    print("\n2️⃣  查看数据库统计...")
    response = requests.get(f"{BASE_URL}/database/stats")
    stats = response.json()

    print(f"   • 会话数: {stats['total_sessions']}")
    print(f"   • Checkpoint数: {stats['total_checkpoints']}")
    print(f"   • 数据库大小: {stats['database_size_mb']} MB")

    # 4. 结果
    print("\n" + "=" * 60)
    if stats['total_checkpoints'] > 0:
        print("✅ 持久化成功！数据已写入SQLite")
        print(f"\n数据库: {stats['database_path']}")
    else:
        print("❌ 持久化失败，数据未写入")
        print("\n请检查:")
        print("1. 查看服务端日志")
        print("2. 确认 aiosqlite 已安装")
    print("=" * 60)

except requests.exceptions.ConnectionError:
    print("\n❌ 无法连接到服务器")
    print("请先启动服务: python main.py")
except Exception as e:
    print(f"\n❌ 错误: {e}")
