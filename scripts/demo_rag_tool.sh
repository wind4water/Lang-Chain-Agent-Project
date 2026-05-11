#!/bin/bash
# RAG 工具集成演示脚本

echo ""
echo "============================================================"
echo "  RAG 工具集成演示"
echo "============================================================"
echo ""

# 检查服务是否运行
echo "1️⃣  检查服务状态..."
response=$(curl -s http://localhost:8000/health)
if [ $? -eq 0 ]; then
    echo "   ✅ 服务正在运行"
    echo "   $response" | python3 -m json.tool 2>/dev/null || echo "   $response"
else
    echo "   ❌ 服务未启动"
    echo "   请先运行: python app/main.py"
    exit 1
fi

echo ""
echo "2️⃣  查看已加载的工具..."
curl -s http://localhost:8000/tools | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"   总共加载了 {data['total']} 个工具\")
for tool in data['tools']:
    marker = '✅' if tool['name'] == 'knowledge_base_search' else '  '
    print(f\"   {marker} {tool['name']}\")
"

echo ""
echo "3️⃣  检查 RAG 系统状态..."
curl -s http://localhost:8000/rag/stats | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data.get('initialized'):
    print(f\"   ✅ RAG 已初始化\")
    print(f\"   文档数量: {data.get('document_count', 0)}\")
    print(f\"   嵌入模型: {data.get('embedding_model', 'N/A')}\")
else:
    print(f\"   ⚠️  RAG 未初始化\")
"

echo ""
echo "4️⃣  测试 Agent 自动调用 RAG 工具..."
echo ""
echo "   问题: '系统支持哪些文档格式？'"
echo "   （Agent 会自动判断是否需要使用知识库）"
echo ""

curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "系统支持哪些文档格式？",
    "session_id": "demo_test"
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
response = data.get('response', '')

print('   🤖 Agent 回答:')
print('   ' + '─' * 70)
for line in response.split('\n'):
    print('   ' + line)
print('   ' + '─' * 70)

# 检查是否使用了 RAG
if '知识库回答' in response or '参考来源' in response:
    print('')
    print('   ✅ Agent 自动使用了 RAG 工具！')
else:
    print('')
    print('   ℹ️  Agent 可能没有使用 RAG 工具')
"

echo ""
echo "5️⃣  清理测试会话..."
curl -s -X DELETE http://localhost:8000/history/demo_test > /dev/null
echo "   ✅ 已清理"

echo ""
echo "============================================================"
echo "  演示完成！"
echo "============================================================"
echo ""
echo "💡 要点："
echo "   - Agent 会根据问题内容自动决定是否使用 RAG"
echo "   - 使用单一的 /chat 接口即可，无需手动判断"
echo "   - RAG 工具可以与其他工具（计算器、搜索等）组合使用"
echo ""
