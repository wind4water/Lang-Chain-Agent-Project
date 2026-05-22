"""
评估切块效果的脚本
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.system import rag_system
from app.rag.config import RAGConfig


def evaluate_chunks():
    """评估切块效果"""
    # 初始化 RAG 系统
    config = RAGConfig()
    rag = RAGSystem()
    rag.initialize(config)
    
    # 获取统计
    stats = rag.get_stats()
    registry = stats.get('registry', {})
    
    total_docs = registry.get('total_documents', 0)
    total_chunks = registry.get('total_chunks', 0)
    total_size = registry.get('total_size_bytes', 0)
    chunk_size = stats.get('config', {}).get('chunk_size', 1000)
    chunk_overlap = stats.get('config', {}).get('chunk_overlap', 200)
    
    print("=" * 60)
    print("📊 切块效果评估报告")
    print("=" * 60)
    print(f"\n📁 文档统计:")
    print(f"   文档总数: {total_docs}")
    print(f"   切块总数: {total_chunks}")
    print(f"   总大小: {registry.get('total_size_mb', 0):.2f} MB")
    
    print(f"\n⚙️  切分配置:")
    print(f"   Chunk Size: {chunk_size}")
    print(f"   Chunk Overlap: {chunk_overlap}")
    
    if total_docs > 0 and total_chunks > 0:
        avg_chunks_per_doc = total_chunks / total_docs
        avg_chunk_size = total_size / total_chunks
        utilization = avg_chunk_size / chunk_size
        
        print(f"\n📈 切块效果:")
        print(f"   每文档平均切块数: {avg_chunks_per_doc:.2f}")
        print(f"   每块平均字符数: {avg_chunk_size:.0f}")
        print(f"   块利用率: {utilization:.2%}")
        
        print(f"\n🎯 评估建议:")
        if avg_chunks_per_doc < 2:
            print("   ⚠️  切块过少，可能导致检索粒度太粗")
        elif avg_chunks_per_doc > 20:
            print("   ⚠️  切块过多，可能影响检索效率")
        else:
            print("   ✅ 切块数量合理")
            
        if utilization < 0.5:
            print("   ⚠️  块利用率低，建议减小 chunk_size 或检查短文档")
        elif utilization > 0.95:
            print("   ⚠️  块利用率高，可能存在大段落被强制截断")
        else:
            print("   ✅ 块利用率合理")
            
        # 预估重叠开销
        overlap_overhead = (chunk_overlap * total_chunks) / (chunk_size * total_chunks)
        print(f"\n📊 重叠开销: {overlap_overhead:.2%}")
    else:
        print("\n⚠️  暂无文档数据，请先添加文档到 data/documents/")


if __name__ == "__main__":
    evaluate_chunks()
