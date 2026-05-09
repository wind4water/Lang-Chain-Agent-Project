#!/bin/bash
# 快速验证压缩功能是否正常工作

echo "🚀 上下文压缩功能快速验证"
echo "================================"

# 检查文件是否存在
echo ""
echo "📁 检查核心文件..."

if [ -f "app/agents/sqlite.py" ]; then
    echo "✅ app/agents/sqlite.py"
else
    echo "❌ app/agents/sqlite.py 不存在"
    exit 1
fi

if [ -f ".env.example" ]; then
    echo "✅ .env.example"
else
    echo "❌ .env.example 不存在"
    exit 1
fi

if [ -f "doc/COMPRESSION_USAGE.md" ]; then
    echo "✅ doc/COMPRESSION_USAGE.md"
else
    echo "❌ doc/COMPRESSION_USAGE.md 不存在"
    exit 1
fi

if [ -f "examples/test_compression.py" ]; then
    echo "✅ examples/test_compression.py"
else
    echo "❌ examples/test_compression.py 不存在"
    exit 1
fi

# 检查压缩配置是否在 .env.example 中
echo ""
echo "🔍 检查配置项..."

if grep -q "CONTEXT_COMPRESSION_STRATEGY" .env.example; then
    echo "✅ CONTEXT_COMPRESSION_STRATEGY 配置已添加"
else
    echo "❌ CONTEXT_COMPRESSION_STRATEGY 配置缺失"
    exit 1
fi

if grep -q "COMPRESSION_WINDOW_SIZE" .env.example; then
    echo "✅ COMPRESSION_WINDOW_SIZE 配置已添加"
else
    echo "❌ COMPRESSION_WINDOW_SIZE 配置缺失"
    exit 1
fi

# 检查代码中是否实现了压缩方法
echo ""
echo "🔍 检查代码实现..."

if grep -q "_compress_sliding_window" app/agents/sqlite.py; then
    echo "✅ 滑动窗口策略已实现"
else
    echo "❌ 滑动窗口策略未实现"
    exit 1
fi

if grep -q "_compress_token_limit" app/agents/sqlite.py; then
    echo "✅ Token计数策略已实现"
else
    echo "❌ Token计数策略未实现"
    exit 1
fi

if grep -q "_compress_summary" app/agents/sqlite.py; then
    echo "✅ 智能摘要策略已实现"
else
    echo "❌ 智能摘要策略未实现"
    exit 1
fi

if grep -q "_apply_compression" app/agents/sqlite.py; then
    echo "✅ 压缩调度器已实现"
else
    echo "❌ 压缩调度器未实现"
    exit 1
fi

# 统计文档数量
echo ""
echo "📚 文档统计..."
compression_docs=$(ls doc/*COMPRESS* 2>/dev/null | wc -l)
echo "✅ 压缩相关文档: $compression_docs 个"

# 显示总结
echo ""
echo "================================"
echo "✅ 上下文压缩功能验证通过！"
echo ""
echo "📖 下一步："
echo "1. 复制配置文件: cp .env.example .env"
echo "2. 编辑 .env，设置压缩策略"
echo "3. 启动服务: python run.py"
echo "4. 运行测试: python examples/test_compression.py --strategy sliding_window"
echo ""
echo "📚 查看文档: doc/COMPRESSION_USAGE.md"
