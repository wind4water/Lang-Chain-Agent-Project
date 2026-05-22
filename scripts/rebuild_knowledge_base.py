import asyncio
import os
import sys
import logging
from dotenv import load_dotenv

# 配置日志输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from app.rag import rag_system

async def main():
    print("🔄 开始全量重建知识库以同步新增的 doc_group 元数据字段...")
    await rag_system.initialize()
    
    # 打印初始统计
    stats = rag_system.get_stats()
    print(f"📋 当前文档总数 (Chunks): {stats.get('document_count', 0)}")
    
    # 开始重建
    result = await rag_system.rebuild_knowledge_base()
    print("\n🎉 知识库全量重建完成！")
    print(f"📋 新文档总数 (Chunks): {result.get('split_documents', 0)}")
    print(f"📋 注册的原文件数: {result.get('registered_files', 0)}")

if __name__ == "__main__":
    asyncio.run(main())
