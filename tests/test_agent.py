"""
LangChain Agent测试脚本
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_chat(message: str, session_id: str = "test_user"):
    """测试对话接口"""
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"message": message, "session_id": session_id}
    )
    return response.json()


def test_history(session_id: str = "test_user"):
    """测试获取历史"""
    response = requests.get(f"{BASE_URL}/history/{session_id}")
    return response.json()


def test_clear_history(session_id: str = "test_user"):
    """测试清除历史"""
    response = requests.delete(f"{BASE_URL}/history/{session_id}")
    return response.json()


def main():
    print("🚀 开始测试LangChain对话Agent\n")

    # 测试1：第一轮对话
    print("=" * 50)
    print("测试1: 自我介绍")
    result = test_chat("你好，我叫小明，很高兴认识你！")
    print(f"AI回复: {result['response']}\n")

    # 测试2：第二轮对话（测试checkpoint记忆）
    print("=" * 50)
    print("测试2: 测试记忆功能")
    result = test_chat("我刚才告诉你我叫什么名字？")
    print(f"AI回复: {result['response']}\n")

    # 测试3：获取历史记录
    print("=" * 50)
    print("测试3: 获取对话历史")
    history = test_history()
    print(f"历史记录 ({len(history['history'])} 条消息):")
    for i, msg in enumerate(history['history'], 1):
        print(f"  {i}. [{msg['role']}]: {msg['content']}")
    print()

    # 测试4：多轮对话
    print("=" * 50)
    print("测试4: 多轮对话")
    questions = [
        "请记住这个数字：42",
        "我刚才让你记住的数字是多少？",
        "把这个数字乘以2是多少？"
    ]
    for q in questions:
        result = test_chat(q)
        print(f"用户: {q}")
        print(f"AI: {result['response']}\n")

    # 测试5：多会话隔离
    print("=" * 50)
    print("测试5: 多会话隔离")
    result1 = test_chat("我喜欢蓝色", session_id="user_1")
    print(f"[user_1] AI回复: {result1['response']}")

    result2 = test_chat("我喜欢红色", session_id="user_2")
    print(f"[user_2] AI回复: {result2['response']}")

    result1 = test_chat("我喜欢什么颜色？", session_id="user_1")
    print(f"[user_1] AI回复: {result1['response']}")

    result2 = test_chat("我喜欢什么颜色？", session_id="user_2")
    print(f"[user_2] AI回复: {result2['response']}\n")

    # 测试6：清除历史
    print("=" * 50)
    print("测试6: 清除历史记录")
    clear_result = test_clear_history()
    print(f"清除结果: {clear_result['message']}")

    result = test_chat("我之前说我叫什么？")
    print(f"AI回复: {result['response']}\n")

    print("=" * 50)
    print("✅ 测试完成！")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("❌ 错误: 无法连接到服务器")
        print("请确保服务已启动: python main.py")
    except Exception as e:
        print(f"❌ 测试出错: {e}")
