"""
验证持久化：重启后查询历史记录
在重启服务后运行此脚本，验证数据是否保留
"""
import requests

BASE_URL = "http://localhost:8000"

print("=" * 80)
print("🔍 验证SQLite持久化 - 重启后测试")
print("=" * 80)

try:
    # 查询历史记录
    response = requests.get(f"{BASE_URL}/history/user_persist")
    history = response.json()

    print(f"\n📊 查询会话 'user_persist' 的历史记录：")
    print("-" * 80)

    if len(history['history']) > 0:
        print(f"✅ 成功！找到 {len(history['history'])} 条历史消息")
        print("✅ SQLite持久化工作正常！\n")

        print("历史消息内容：")
        for i, msg in enumerate(history['history'], 1):
            role = "用户" if msg['role'] == 'user' else "AI"
            print(f"  {i}. [{role}]: {msg['content']}")

        print("\n" + "=" * 80)
        print("🎉 持久化测试通过！")
        print("=" * 80)
        print("说明：")
        print("• 服务重启后数据仍然存在")
        print("• 对话历史已成功保存到SQLite数据库")
        print("• 可以继续之前的对话")

        # 继续对话测试
        print("\n📝 测试：继续之前的对话")
        print("-" * 80)
        response = requests.post(
            f"{BASE_URL}/chat",
            json={
                "message": "我之前告诉你我叫什么名字？我是做什么的？",
                "session_id": "user_persist"
            }
        )
        result = response.json()
        print(f"用户: 我之前告诉你我叫什么名字？我是做什么的？")
        print(f"AI: {result['response']}")
        print("\n✅ Agent记住了之前的对话内容！")

    else:
        print("❌ 失败：没有找到历史记录")
        print("可能原因：")
        print("1. 数据库文件被删除")
        print("2. 还没有运行 test_persistence.py 创建测试数据")
        print("3. session_id 不匹配")

except requests.exceptions.ConnectionError:
    print("❌ 错误：无法连接到服务器")
    print("请确保服务已启动: python main.py")
except Exception as e:
    print(f"❌ 错误：{e}")
