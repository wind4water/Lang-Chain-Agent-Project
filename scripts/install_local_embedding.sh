#!/bin/bash

# 本地 Embedding 快速安装脚本

echo "=========================================="
echo "安装本地 Embedding 依赖"
echo "=========================================="
echo ""

echo "正在安装必要的依赖包..."
echo "  - langchain-huggingface: LangChain 与 HuggingFace 集成"
echo "  - sentence-transformers: 本地向量化模型"
echo "  - torch: PyTorch（模型运行依赖）"
echo ""

pip install langchain-huggingface sentence-transformers torch

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 依赖安装成功！"
    echo ""
    echo "=========================================="
    echo "配置本地 Embedding"
    echo "=========================================="
    echo ""
    echo "请在 .env 文件中配置："
    echo ""
    echo "  EMBEDDING_TYPE=local"
    echo "  EMBEDDING_MODEL=bge-base-zh-v1.5"
    echo "  EMBEDDING_DEVICE=cpu"
    echo ""
    echo "=========================================="
    echo "测试本地 Embedding"
    echo "=========================================="
    echo ""
    echo "运行测试脚本："
    echo "  python tests/test_local_embedding.py"
    echo ""
    echo "首次运行会自动下载模型（约 400MB），请耐心等待。"
    echo "模型会缓存到: ~/.cache/huggingface/hub/"
    echo ""
else
    echo ""
    echo "❌ 依赖安装失败！"
    echo ""
    echo "请检查："
    echo "  1. Python 版本是否 >= 3.8"
    echo "  2. pip 是否正常工作"
    echo "  3. 网络连接是否正常"
    echo ""
fi
