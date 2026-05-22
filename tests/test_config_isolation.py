"""
测试配置隔离功能 - 不同 embedding 配置使用独立目录

演示：
1. 使用本地模型 bge-base-zh-v1.5 建立索引
2. 切换到远程 API text-embedding-3-small（模拟）
3. 再切回本地模型（无需重建）
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.rag.config import RAGConfig


def test_config_isolation():
    """测试配置隔离机制"""
    print("=" * 70)
    print("配置隔离功能测试")
    print("=" * 70)
    print()

    # 场景 1: 本地模型 bge-base-zh-v1.5
    print("场景 1: 本地模型 bge-base-zh-v1.5")
    print("-" * 70)
    os.environ["EMBEDDING_TYPE"] = "local"
    os.environ["EMBEDDING_MODEL"] = "bge-base-zh-v1.5"

    config1 = RAGConfig()
    vector_path1 = config1.get_vector_store_path()
    registry_path1 = config1.get_registry_path()

    print(f"  EMBEDDING_TYPE: {config1.embedding_type}")
    print(f"  EMBEDDING_MODEL: {config1.embedding_model}")
    print(f"  向量库路径: {vector_path1}")
    print(f"  注册表路径: {registry_path1}")
    print()

    # 场景 2: 远程 API text-embedding-3-small
    print("场景 2: 远程 API text-embedding-3-small")
    print("-" * 70)
    os.environ["EMBEDDING_TYPE"] = "remote"
    os.environ["EMBEDDING_MODEL"] = "text-embedding-3-small"

    config2 = RAGConfig()
    vector_path2 = config2.get_vector_store_path()
    registry_path2 = config2.get_registry_path()

    print(f"  EMBEDDING_TYPE: {config2.embedding_type}")
    print(f"  EMBEDDING_MODEL: {config2.embedding_model}")
    print(f"  向量库路径: {vector_path2}")
    print(f"  注册表路径: {registry_path2}")
    print()

    # 场景 3: 另一个本地模型 all-MiniLM-L6-v2
    print("场景 3: 本地模型 all-MiniLM-L6-v2")
    print("-" * 70)
    os.environ["EMBEDDING_TYPE"] = "local"
    os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"

    config3 = RAGConfig()
    vector_path3 = config3.get_vector_store_path()
    registry_path3 = config3.get_registry_path()

    print(f"  EMBEDDING_TYPE: {config3.embedding_type}")
    print(f"  EMBEDDING_MODEL: {config3.embedding_model}")
    print(f"  向量库路径: {vector_path3}")
    print(f"  注册表路径: {registry_path3}")
    print()

    # 验证路径隔离
    print("✅ 验证结果")
    print("=" * 70)

    # 检查三个配置的路径是否完全不同
    paths = [vector_path1, vector_path2, vector_path3]
    registries = [registry_path1, registry_path2, registry_path3]

    if len(set(paths)) == 3:
        print("✅ 向量库路径完全隔离（3个不同路径）")
    else:
        print("❌ 向量库路径有重复！")
        return False

    if len(set(registries)) == 3:
        print("✅ 注册表路径完全隔离（3个不同文件）")
    else:
        print("❌ 注册表路径有重复！")
        return False

    print()
    print("📊 路径对比")
    print("-" * 70)
    for i, (vp, rp) in enumerate(zip(paths, registries), 1):
        print(f"配置 {i}:")
        print(f"  向量库: {vp}")
        print(f"  注册表: {rp}")
        print()

    print("=" * 70)
    print("✨ 配置隔离功能正常工作！")
    print("=" * 70)
    print()
    print("使用说明：")
    print("1. 修改 .env 中的 EMBEDDING_TYPE 和 EMBEDDING_MODEL")
    print("2. 重启服务 - 自动加载对应配置的向量库")
    print("3. 无需重建索引，切换即用")
    print("4. 可以同时保留多个配置的向量库进行对比")
    print()

    return True


if __name__ == "__main__":
    try:
        success = test_config_isolation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
