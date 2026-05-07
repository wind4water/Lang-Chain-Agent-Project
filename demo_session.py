"""
演示session_id的上下文隔离机制
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def chat(message: str, session_id: str):
    """发送消息"""
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"message": message, "session_id": session_id}
    )
    result = response.json()
    print(f"[{session_id}] 用户: {message}")
    print(f"[{session_id}] AI: {result['response']}\n")
    return result['response']


print("=" * 80)
print("演示：相同session_id = 相同上下文")
print("=" * 80)

# 用户A的第一轮对话
chat("我叫张三，今年25岁", session_id="user_A")

# 用户A的第二轮对话 - 会记住之前说的内容
chat("我叫什么名字？多大了？", session_id="user_A")

print("\n" + "=" * 80)
print("演示：不同session_id = 独立上下文（互不干扰）")
print("=" * 80)

# 用户B的第一轮对话
chat("我叫李四，我喜欢蓝色", session_id="user_B")

# 用户C的第一轮对话
chat("我叫王五，我喜欢红色", session_id="user_C")

# 用户B询问自己的信息 - 只会记住user_B的内容
chat("我叫什么？我喜欢什么颜色？", session_id="user_B")

# 用户C询问自己的信息 - 只会记住user_C的内容
chat("我叫什么？我喜欢什么颜色？", session_id="user_C")

# 用户A再次对话 - 仍然记住最开始的内容
chat("我之前说我多大了？", session_id="user_A")

print("\n" + "=" * 80)
print("演示：查看各个会话的历史记录")
print("=" * 80)

for session in ["user_A", "user_B", "user_C"]:
    response = requests.get(f"{BASE_URL}/history/{session}")
    history = response.json()
    print(f"\n【{session}的对话历史】共{len(history['history'])}条消息：")
    for i, msg in enumerate(history['history'], 1):
        role = "用户" if msg['role'] == 'user' else "AI"
        print(f"  {i}. {role}: {msg['content'][:50]}...")

print("\n" + "=" * 80)
print("✅ 演示完成！")
print("=" * 80)
print("\n总结：")
print("• 相同session_id → 共享上下文，可以进行连续对话")
print("• 不同session_id → 完全隔离，互不影响")
print("• 适合多用户、多会话场景（如客服系统、聊天机器人等）")
