"""
测试SQLite数据库查询接口
演示如何查看数据库中存储的所有数据
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_sessions_list():
    """测试：获取所有会话列表"""
    print_section("1️⃣  查询所有会话ID")
    response = requests.get(f"{BASE_URL}/sessions")
    data = response.json()

    print(f"\n总会话数: {data['total']}")
    if data['sessions']:
        print("\n会话列表:")
        for i, session in enumerate(data['sessions'], 1):
            print(f"  {i}. {session}")
    else:
        print("\n暂无会话数据")

    return data


def test_database_stats():
    """测试：获取数据库统计信息"""
    print_section("2️⃣  数据库统计信息")
    response = requests.get(f"{BASE_URL}/database/stats")
    data = response.json()

    print(f"\n📊 总体统计:")
    print(f"  • 总会话数: {data['total_sessions']}")
    print(f"  • 总checkpoint数: {data['total_checkpoints']}")
    print(f"  • 数据库大小: {data['database_size_mb']} MB ({data['database_size_bytes']} 字节)")
    print(f"  • 数据库路径: {data['database_path']}")

    if data['sessions']:
        print(f"\n📋 会话详情 (按最后活跃时间排序):")
        print(f"  {'Session ID':<30} {'Checkpoints':<15} {'最后活跃':<20}")
        print(f"  {'-'*30} {'-'*15} {'-'*20}")
        for session in data['sessions']:
            print(f"  {session['session_id']:<30} {session['checkpoint_count']:<15} {session['last_seen']:<20}")

    return data


def test_session_detail(session_id):
    """测试：获取指定会话的详细信息"""
    print_section(f"3️⃣  会话详细信息: {session_id}")
    response = requests.get(f"{BASE_URL}/database/sessions/{session_id}")
    data = response.json()

    print(f"\n会话ID: {data['session_id']}")
    print(f"Checkpoint总数: {data['checkpoint_count']}")

    if data['checkpoints']:
        print(f"\nCheckpoint列表 (最新的在前):")
        print(f"  {'Checkpoint ID':<40} {'时间戳':<25} {'有数据'}")
        print(f"  {'-'*40} {'-'*25} {'-'*6}")
        for cp in data['checkpoints'][:5]:  # 只显示最新的5条
            has_data = "✅" if cp['has_data'] else "❌"
            print(f"  {cp['checkpoint_id']:<40} {cp['timestamp']:<25} {has_data}")

        if len(data['checkpoints']) > 5:
            print(f"  ... 还有 {len(data['checkpoints']) - 5} 条记录")

    return data


def test_session_history(session_id):
    """测试：获取会话对话历史"""
    print_section(f"4️⃣  会话对话历史: {session_id}")
    response = requests.get(f"{BASE_URL}/history/{session_id}")
    data = response.json()

    if data['history']:
        print(f"\n对话记录 (共 {len(data['history'])} 条消息):")
        for i, msg in enumerate(data['history'], 1):
            role = "用户" if msg['role'] == 'user' else "AI"
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            print(f"\n  {i}. [{role}]: {content}")
    else:
        print("\n暂无对话记录")

    return data


def main():
    print("🔍 SQLite数据库查询接口测试")

    try:
        # 1. 先创建一些测试数据（如果没有的话）
        print_section("0️⃣  准备测试数据")
        sessions = requests.get(f"{BASE_URL}/sessions").json()

        if sessions['total'] == 0:
            print("\n当前没有数据，创建一些测试数据...")

            # 创建几个测试会话
            test_data = [
                ("user_alice", "你好，我是Alice"),
                ("user_bob", "我是Bob，很高兴认识你"),
                ("user_charlie", "Charlie在这里")
            ]

            for session_id, message in test_data:
                requests.post(
                    f"{BASE_URL}/chat",
                    json={"message": message, "session_id": session_id}
                )
                print(f"  ✓ 创建会话: {session_id}")

        # 2. 获取所有会话列表
        sessions_data = test_sessions_list()

        # 3. 获取数据库统计信息
        stats_data = test_database_stats()

        # 4. 如果有会话，查看第一个会话的详细信息
        if sessions_data['sessions']:
            first_session = sessions_data['sessions'][0]
            test_session_detail(first_session)
            test_session_history(first_session)

        # 5. 总结
        print_section("✅ 测试完成")
        print(f"""
新增接口说明：

1️⃣  GET /sessions
   • 返回所有会话ID列表
   • 用于查看系统中有哪些用户/会话

2️⃣  GET /database/stats
   • 返回数据库统计信息
   • 包含总会话数、checkpoint数、数据库大小
   • 每个会话的详细统计（checkpoint数、活跃时间）

3️⃣  GET /database/sessions/{{session_id}}
   • 返回指定会话的底层checkpoint信息
   • 用于调试和了解数据存储结构

4️⃣  GET /history/{{session_id}} (已有)
   • 返回会话的对话历史
   • 用户友好的格式

使用示例：
  # 查看所有会话
  curl http://localhost:8000/sessions

  # 查看数据库统计
  curl http://localhost:8000/database/stats

  # 查看指定会话详情
  curl http://localhost:8000/database/sessions/user_alice

  # 查看对话历史
  curl http://localhost:8000/history/user_alice
""")

    except requests.exceptions.ConnectionError:
        print("\n❌ 错误：无法连接到服务器")
        print("请确保服务已启动: python main.py")
    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
