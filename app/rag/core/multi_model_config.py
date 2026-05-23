"""
多模型配置管理
支持路径级别的模型配置
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class PathModelConfig:
    """路径模型配置"""
    path: str                      # 路径（可以是完整路径或包含关键词）
    model_type: str               # 模型类型: bge | codebert | openai
    model_name: str               # 具体模型名称
    dimension: int                # 向量维度
    description: str              # 配置描述

    def matches(self, file_path: str) -> bool:
        """检查文件路径是否匹配此配置"""
        # 标准化路径：移除 ./ 前缀，统一使用相对路径格式
        normalized_path = file_path.lstrip("./")
        normalized_self_path = self.path.lstrip("./")
        
        # 支持完整路径匹配或包含关键词匹配
        if normalized_self_path in normalized_path:
            return True
        # 支持通配符匹配（简单实现）
        import fnmatch
        if fnmatch.fnmatch(normalized_path, normalized_self_path):
            return True
        return False


class MultiModelConfig:
    """
    多模型配置管理器

    支持为不同路径配置不同的嵌入模型
    示例配置：
    - ./data/documents → BGE（通用文本）
    - ./data/projects → CodeBERT（代码）
    - ./data/projects/docs → BGE（项目文档）
    """

    def __init__(self, config_str: str = None):
        """
        初始化多模型配置

        Args:
            config_str: JSON 格式的配置字符串
        """
        self.path_configs: List[PathModelConfig] = []
        self.default_config = PathModelConfig(
            path="*",
            model_type="bge",
            model_name="BAAI/bge-base-zh-v1.5",
            dimension=768,
            description="默认 BGE 模型"
        )

        if config_str:
            self._parse_config(config_str)

    def _parse_config(self, config_str: str):
        """解析配置字符串"""
        try:
            # 尝试解析为新的嵌套格式
            data = json.loads(config_str)
            if "configs" in data:
                configs = data["configs"]
            else:
                configs = data if isinstance(data, list) else [data]
            
            for cfg in configs:
                self.path_configs.append(PathModelConfig(
                    path=cfg["path"],
                    model_type=cfg["model_type"],
                    model_name=cfg["model_name"],
                    dimension=cfg["dimension"],
                    description=cfg.get("description", "")
                ))
            logger.info(f"加载了 {len(self.path_configs)} 条路径模型配置")
        except Exception as e:
            logger.error(f"解析多模型配置失败: {e}")
            logger.info("使用默认配置")

    def get_model_for_path(self, file_path: str) -> PathModelConfig:
        """
        根据文件路径获取对应的模型配置

        Args:
            file_path: 文件路径

        Returns:
            匹配的模型配置
        """
        # 按路径长度排序（长的优先，更精确）
        sorted_configs = sorted(
            self.path_configs,
            key=lambda x: len(x.path),
            reverse=True
        )

        for config in sorted_configs:
            if config.matches(file_path):
                logger.debug(f"路径 {file_path} 匹配模型: {config.model_name}")
                return config

        return self.default_config

    def get_all_paths(self) -> List[str]:
        """获取所有配置的路径列表"""
        return [cfg.path for cfg in self.path_configs]

    def get_unique_models(self) -> Dict[str, PathModelConfig]:
        """获取所有唯一的模型配置（按 model_name 去重）"""
        models = {}
        for cfg in self.path_configs:
            models[cfg.model_name] = cfg
        return models


# 默认配置示例（JSON格式）
DEFAULT_MULTI_MODEL_CONFIG = """
[
    {
        "path": "./data/documents",
        "model_type": "bge",
        "model_name": "BAAI/bge-base-zh-v1.5",
        "dimension": 768,
        "description": "通用文档，使用 BGE 模型"
    },
    {
        "path": "./data/projects",
        "model_type": "unixcoder",
        "model_name": "microsoft/unixcoder-base",
        "dimension": 768,
        "description": "代码项目，使用 UniXcoder 模型（专为代码检索优化）"
    },
    {
        "path": "./data/projects/*.md",
        "model_type": "bge",
        "model_name": "BAAI/bge-base-zh-v1.5",
        "dimension": 768,
        "description": "项目中的 Markdown 文档，使用 BGE 模型"
    }
]
"""
