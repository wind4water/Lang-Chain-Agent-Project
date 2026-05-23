"""
测试 UniXcoder 嵌入模型
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO)


def test_unixcoder_embeddings():
    """测试 UniXcoder 嵌入"""
    print("=" * 60)
    print("测试 UniXcoder 嵌入模型")
    print("=" * 60)
    
    try:
        from app.rag.core.code_embeddings import UniXcoderEmbeddings
        
        print("\n1. 创建 UniXcoder 嵌入模型...")
        embeddings = UniXcoderEmbeddings(
            model_name="microsoft/unixcoder-base",
            device="cpu"
        )
        print("✅ 模型创建成功")
        
        print("\n2. 测试代码嵌入...")
        code_samples = [
            "public enum LanguageEnum { CN, EN, JP }",
            "public class User { private String name; }",
            "def hello_world(): print('Hello')"
        ]
        
        vectors = embeddings.embed_documents(code_samples)
        print(f"✅ 嵌入 {len(code_samples)} 个代码片段")
        print(f"   向量维度: {len(vectors[0])}")
        
        print("\n3. 测试查询嵌入...")
        query = "LanguageEnum 有哪些值"
        query_vector = embeddings.embed_query(query)
        print(f"✅ 查询向量生成成功")
        print(f"   向量维度: {len(query_vector)}")
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_model_config():
    """测试多模型配置"""
    print("\n" + "=" * 60)
    print("测试多模型配置")
    print("=" * 60)
    
    try:
        from app.rag.core.multi_model_config import MultiModelConfig, PathModelConfig
        
        # 使用新的配置
        config_json = '{"configs":[{"path":"./data/documents","model_type":"bge","model_name":"BAAI/bge-base-zh-v1.5","dimension":768},{"path":"./data/projects","model_type":"unixcoder","model_name":"microsoft/unixcoder-base","dimension":768}]}'
        
        config = MultiModelConfig(config_json)
        print(f"✅ 配置加载成功")
        print(f"   配置数量: {len(config.path_configs)}")
        
        for cfg in config.path_configs:
            print(f"   - {cfg.path}: {cfg.model_type} ({cfg.model_name})")
        
        # 测试路径匹配
        test_paths = [
            "./data/documents/readme.md",
            "./data/projects/test.java",
            "./data/projects/README.md"
        ]
        
        print("\n路径匹配测试:")
        for path in test_paths:
            model_cfg = config.get_model_for_path(path)
            print(f"   {path} -> {model_cfg.model_type} ({model_cfg.model_name})")
        
        print("\n" + "=" * 60)
        print("✅ 配置测试通过！")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行测试
    success1 = test_multi_model_config()
    success2 = test_unixcoder_embeddings()
    
    if success1 and success2:
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！UniXcoder 集成准备就绪。")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("⚠️ 部分测试失败，请检查错误信息。")
        print("=" * 60)
        sys.exit(1)
