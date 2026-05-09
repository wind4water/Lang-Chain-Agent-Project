"""
测试新功能：工具库扩展、Token统计、流式响应

运行前请确保服务已启动：python app/main.py
"""

import requests
import json
import time


BASE_URL = "http://localhost:8000"
SESSION_ID = f"test_{int(time.time())}"


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def test_new_tools():
    """测试新增的6个工具"""
    print_section("测试1: 新增工具")

    # 1. 测试计算器
    print("1️⃣ 测试计算器工具...")
    response = requests.post(f"{BASE_URL}/chat", json={
        "message": "帮我计算 (125 + 75) * 3",
        "session_id": SESSION_ID
    })
    print(f"问题: 帮我计算 (125 + 75) * 3")
    print(f"回答: {response.json()['response']}\n")

    # 2. 测试天气查询（可能需要等待）
    print("2️⃣ 测试天气查询...")
    response = requests.post(f"{BASE_URL}/chat", json={
        "message": "北京今天天气怎么样？",
        "session_id": SESSION_ID
    })
    print(f"问题: 北京今天天气怎么样？")
    print(f"回答: {response.json()['response']}\n")

    # 3. 测试货币转换
    print("3️⃣ 测试货币转换...")
    response = requests.post(f"{BASE_URL}/chat", json={
        "message": "100美元等于多少人民币？",
        "session_id": SESSION_ID
    })
    print(f"问题: 100美元等于多少人民币？")
    print(f"回答: {response.json()['response']}\n")

    # 4. 测试网页抓取
    print("4️⃣ 测试网页内容抓取...")
    response = requests.post(f"{BASE_URL}/chat", json={
        "message": "帮我看看 https://example.com 这个网页的内容",
        "session_id": SESSION_ID
    })
    print(f"问题: 帮我看看 https://example.com 这个网页的内容")
    print(f"回答: {response.json()['response'][:200]}...\n")

    # 5. 测试Python代码执行
    print("5️⃣ 测试Python代码执行器...")
    response = requests.post(f"{BASE_URL}/chat", json={
        "message": "帮我用Python计算1到100的和",
        "session_id": SESSION_ID
    })
    print(f"问题: 帮我用Python计算1到100的和")
    print(f"回答: {response.json()['response']}\n")

    print("✅ 所有工具测试完成！")


def test_token_stats():
    """测试Token统计功能"""
    print_section("测试2: Token统计和成本追踪")

    # 先进行几次对话
    print("发送几条测试消息...")
    for i in range(3):
        requests.post(f"{BASE_URL}/chat", json={
            "message": f"这是测试消息 {i+1}，请简单回复",
            "session_id": SESSION_ID
        })
        time.sleep(0.5)

    # 查询会话的Token统计
    print("\n1️⃣ 查询会话Token统计...")
    response = requests.get(f"{BASE_URL}/stats/tokens/{SESSION_ID}")
    stats = response.json()
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # 查询今日统计
    print("\n2️⃣ 查询今日Token统计...")
    response = requests.get(f"{BASE_URL}/stats/tokens/daily")
    daily = response.json()
    print(json.dumps(daily, indent=2, ensure_ascii=False))

    # 查询本月统计
    print("\n3️⃣ 查询本月Token统计...")
    response = requests.get(f"{BASE_URL}/stats/tokens/monthly")
    monthly = response.json()
    print(json.dumps(monthly, indent=2, ensure_ascii=False))

    print("\n✅ Token统计测试完成！")


def test_streaming():
    """测试流式响应"""
    print_section("测试3: 流式响应（SSE）")

    print("发送流式请求...")
    print("问题: 请给我讲一个简短的故事\n")
    print("AI回复（流式输出）:")
    print("-" * 60)

    # 发送流式请求
    with requests.post(
        f"{BASE_URL}/chat/stream",
        json={
            "message": "请给我讲一个简短的故事",
            "session_id": SESSION_ID + "_stream"
        },
        stream=True
    ) as response:
        full_text = ""
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # 去掉 "data: " 前缀

                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        if 'content' in data:
                            content = data['content']
                            print(content, end='', flush=True)
                            full_text += content
                        elif 'error' in data:
                            print(f"\n错误: {data['error']}")
                            break
                    except json.JSONDecodeError:
                        pass

    print("\n" + "-" * 60)
    print(f"\n总字符数: {len(full_text)}")
    print("✅ 流式响应测试完成！")


def test_tools_list():
    """测试工具列表"""
    print_section("测试4: 查看所有可用工具")

    response = requests.get(f"{BASE_URL}/tools")
    tools = response.json()

    print(f"工具系统状态: {'启用' if tools['enabled'] else '禁用'}")
    print(f"可用工具数量: {tools['total']}\n")

    print("工具列表:")
    for i, tool in enumerate(tools['tools'], 1):
        print(f"{i}. {tool['name']}")
        print(f"   描述: {tool['description']}\n")

    print("✅ 工具列表查询完成！")


def main():
    """运行所有测试"""
    print("\n" + "🚀" * 30)
    print("  LangChain Agent 新功能测试")
    print("🚀" * 30)

    try:
        # 检查服务是否运行
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("❌ 服务未正常运行，请先启动: python app/main.py")
            return
    except requests.exceptions.RequestException:
        print("❌ 无法连接到服务，请先启动: python app/main.py")
        return

    print(f"\n✅ 服务运行正常")
    print(f"📝 测试会话ID: {SESSION_ID}\n")

    try:
        # 测试1: 新工具
        test_tools_list()
        test_new_tools()

        # 测试2: Token统计
        test_token_stats()

        # 测试3: 流式响应
        test_streaming()

        print("\n" + "🎉" * 30)
        print("  所有测试完成！")
        print("🎉" * 30 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
