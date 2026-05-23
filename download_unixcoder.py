"""
下载 UniXcoder 模型脚本
"""
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_unixcoder():
    """下载 UniXcoder 模型"""
    model_name = "microsoft/unixcoder-base"
    
    logger.info(f"开始下载 UniXcoder 模型: {model_name}")
    logger.info("这可能需要几分钟，请耐心等待...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        logger.info("正在下载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=False
        )
        logger.info("✅ 分词器下载完成")
        
        logger.info("正在下载模型...")
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=False
        )
        logger.info("✅ 模型下载完成")
        
        logger.info("=" * 50)
        logger.info("UniXcoder 模型下载成功！")
        logger.info(f"模型名称: {model_name}")
        logger.info(f"向量维度: 768")
        logger.info("=" * 50)
        
        return True
        
    except ImportError as e:
        logger.error(f"缺少依赖包: {e}")
        logger.error("请运行: pip install transformers torch")
        return False
    except Exception as e:
        logger.error(f"下载失败: {e}")
        return False


if __name__ == "__main__":
    success = download_unixcoder()
    sys.exit(0 if success else 1)
