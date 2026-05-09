#!/usr/bin/env python3
"""
LangChain 对话 Agent 启动脚本

使用方式：
    python run.py              # 启动服务（默认SQLite）
    python run.py --memory     # 使用内存存储
    python run.py --postgres   # 使用PostgreSQL
    python run.py --port 8080  # 指定端口
"""
import argparse
import uvicorn
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description='启动LangChain对话Agent服务')

    # 存储后端选择
    storage_group = parser.add_mutually_exclusive_group()
    storage_group.add_argument(
        '--memory',
        action='store_true',
        help='使用内存存储（重启后数据丢失）'
    )
    storage_group.add_argument(
        '--postgres',
        action='store_true',
        help='使用PostgreSQL存储'
    )
    storage_group.add_argument(
        '--sqlite',
        action='store_true',
        default=True,
        help='使用SQLite存储（默认）'
    )

    # 服务配置
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='服务器地址（默认：0.0.0.0）'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='服务器端口（默认：8000）'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        default=True,
        help='启用热重载（默认：启用）'
    )
    parser.add_argument(
        '--no-reload',
        action='store_true',
        help='禁用热重载'
    )

    args = parser.parse_args()

    # 确定使用的存储后端
    if args.memory:
        storage = 'memory'
        print("🧠 使用内存存储（MemorySaver）")
        print("   ⚠️  重启后数据会丢失")
    elif args.postgres:
        storage = 'postgres'
        print("🐘 使用PostgreSQL存储")
        print("   ⚠️  确保PostgreSQL服务已启动")
    else:  # sqlite (default)
        storage = 'sqlite'
        print("💾 使用SQLite持久化存储")
        print("   ✅ 数据保存在 checkpoints/conversations.db")

    # 设置环境变量（可选）
    import os
    os.environ['AGENT_STORAGE'] = storage

    # 启动服务
    print(f"\n🚀 启动服务...")
    print(f"   地址: http://{args.host}:{args.port}")
    print(f"   文档: http://{args.host}:{args.port}/docs")
    print(f"   热重载: {'启用' if (args.reload and not args.no_reload) else '禁用'}")
    print(f"\n按 Ctrl+C 停止服务\n")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload and not args.no_reload
    )


if __name__ == "__main__":
    main()
