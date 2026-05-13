"""
飞书机器人配置
"""
import os
from typing import Optional


class FeishuConfig:
    """飞书配置管理"""

    def __init__(self):
        self.enabled = os.getenv("FEISHU_ENABLED", "false").lower() == "true"
        self.app_id = os.getenv("FEISHU_APP_ID", "")
        self.app_secret = os.getenv("FEISHU_APP_SECRET", "")
        # SSL 验证配置（默认启用，连接自签名证书 API 时可禁用）
        self.ssl_verify = os.getenv("FEISHU_SSL_VERIFY", "true").lower() == "true"
        # 流式推送字符阈值，默认 20 字符
        self.stream_chunk_size = int(os.getenv("FEISHU_STREAM_CHUNK_SIZE", "20"))
        # 流式推送间隔（毫秒），默认 300ms
        self.stream_interval_ms = int(os.getenv("FEISHU_STREAM_INTERVAL_MS", "300"))
        # 消息内容最大长度（飞书卡片 markdown 元素限制约 3000 字符）
        self.max_card_content = int(os.getenv("FEISHU_MAX_CARD_CONTENT", "2800"))

    def is_configured(self) -> bool:
        """检查配置是否完整"""
        return self.enabled and bool(self.app_id) and bool(self.app_secret)

    def validate(self) -> Optional[str]:
        """验证配置，返回错误信息或 None"""
        if not self.enabled:
            return None  # 未启用，不需要验证
        if not self.app_id:
            return "FEISHU_APP_ID 未配置"
        if not self.app_secret:
            return "FEISHU_APP_SECRET 未配置"
        return None
