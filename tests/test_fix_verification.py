"""
验证SQLite持久化修复
测试数据是否正确存储到数据库
"""
import requests
import time

BASE_URL = "http://localhost:8000"

def test_chat(message, session_id):
    """发送消息"""
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"message": message, "session_id": session_id}
    )
    return response.json()

def test_stats():
    """查看数据库统计"""
    response = requests.get(f"{BASE_URL}/database/stats")
    return response.json()

def test_sessions():
    """查看所有会话"""
    response = requests.get(f"{BASE_URL}/sessions")
    return response.json()

print("=" * 80)
print("🔧 验证SQLite持久化修复")
print("=" * 80)

try:
    # 1. 检查初始状态
    print("\n1️⃣  检查初始数据库状态...")
    stats = test_stats()
    if "error" in stats:
        print(f"   ❌ 数据库错误: {stats['error']}")
        print("\n   请重启服务: python main.py")
        exit(1)
    else:
        print(f"   ✅ 数据库正常")
        print(f"   • 总会话数: {stats['total_sessions']}")
        print(f"   • 总checkpoint数: {stats['total_checkpoints']}")

    # 2. 发送测试消息
    print("\n2️⃣  发送测试消息...")
    result = test_chat("你好，我叫测试用户", "test_user_001")
    print(f"   ✅ 消息已发送")
    print(f"   AI回复: {result['response'][:50]}...")

    # 3. 等待一下，确保数据已写入
    time.sleep(0.5)

    # 4. 检查数据是否存储
    print("\n3️⃣  检查数据是否已存储...")
    stats = test_stats()
    print(f"   • 总会话数: {stats['total_sessions']}")
    print(f"   • 总checkpoint数: {stats['total_checkpoints']}")
    print(f"   • 数据库大小: {stats['database_size_mb']} MB")

    if stats['total_sessions'] > 0:
        print(f"   ✅ 数据已成功存储！")
    else:
        print(f"   ❌ 数据未存储，可能存在问题")

    # 5. 查看会话列表
    print("\n4️⃣  查看会话列表...")
    sessions = test_sessions()
    print(f"   总会话数: {sessions['total']}")
    if sessions['sessions']:
        print(f"   会话列表:")
        for s in sessions['sessions']:
            print(f"      • {s}")
    else:
        print(f"   ❌ 没有找到会话")

    # 6. 再发送几条消息测试
    print("\n5️⃣  测试多轮对话...")
    test_chat("请记住我叫测试用户", "test_user_001")
    test_chat("我是谁？", "test_user_001")

    # 7. 最终统计
    print("\n6️⃣  最终统计...")
    stats = test_stats()
    print(f"   • 总会话数: {stats['total_sessions']}")
    print(f"   • 总checkpoint数: {stats['total_checkpoints']}")

    if stats['sessions']:
        print(f"\n   会话详情:")
        for s in stats['sessions']:
            print(f"      • {s['session_id']}: {s['checkpoint_count']} checkpoints")

    print("\n" + "=" * 80)
    if stats['total_checkpoints'] > 0:
        print("✅ 持久化修复成功！数据已正确存储到SQLite")
    else:
        print("❌ 持久化仍有问题，数据未存储")
    print("=" * 80)

except requests.exceptions.ConnectionError:
    print("\n❌ 错误：无法连接到服务器")
    print("请确保服务已启动: python main.py")
except Exception as e:
    print(f"\n❌ 错误：{e}")
    import traceback
    traceback.print_exc()
