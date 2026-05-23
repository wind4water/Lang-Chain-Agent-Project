"""
RAG 模块日志工具

提供统一的日志记录接口和装饰器
"""
import logging
import time
import functools
from typing import Optional, Any, Dict


def get_logger(name: str) -> logging.Logger:
    """
    获取 RAG 模块的 logger
    
    Args:
        name: 通常是 __name__
        
    Returns:
        配置好的 logger 实例
    """
    # 确保名称以 app.rag 开头
    if not name.startswith("app.rag"):
        name = f"app.rag.{name}"
    return logging.getLogger(name)


class timed:
    """
    计时上下文管理器和装饰器
    
    用法:
        # 作为上下文管理器
        with timed("文档加载"):
            docs = loader.load()
            
        # 作为装饰器
        @timed("嵌入计算")
        def embed(docs):
            return model.embed(docs)
    """
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None, level: int = logging.INFO):
        self.operation = operation
        self.logger = logger or get_logger("performance")
        self.level = level
        self.start_time: Optional[float] = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(self.level, "[PERF] %s 开始", self.operation)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if exc_type:
            self.logger.error(
                "[PERF] %s 失败 | 耗时: %.3fs | 错误: %s",
                self.operation, elapsed, exc_val
            )
        else:
            self.logger.log(
                self.level,
                "[PERF] %s 完成 | 耗时: %.3fs",
                self.operation, elapsed
            )
        return False
    
    def __call__(self, func):
        """作为装饰器使用"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


def log_structured(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    **kwargs
) -> None:
    """
    输出结构化日志
    
    Args:
        logger: logger 实例
        level: 日志级别
        event: 事件类型（使用 RAGLogEvent 中的常量）
        message: 消息文本
        **kwargs: 额外的结构化数据
    """
    # 构建结构化消息
    parts = [f"[{event}] {message}"]
    
    # 添加结构化数据
    for key, value in kwargs.items():
        if isinstance(value, str) and len(value) > 100:
            value = value[:100] + "..."
        parts.append(f"{key}={value}")
    
    logger.log(level, " | ".join(parts))


def log_step(
    logger: logging.Logger,
    step_number: int,
    total_steps: int,
    description: str,
    status: str = "start"
) -> None:
    """
    记录步骤日志
    
    Args:
        logger: logger 实例
        step_number: 当前步骤序号
        total_steps: 总步骤数
        description: 步骤描述
        status: 状态 (start/complete/failed)
    """
    status_markers = {
        "start": "▶️",
        "complete": "✅",
        "failed": "❌",
        "skip": "⏭️"
    }
    marker = status_markers.get(status, "➡️")
    
    logger.info(
        "[%s] 步骤 %d/%d: %s",
        marker, step_number, total_steps, description
    )


class ProgressLogger:
    """进度日志记录器"""
    
    def __init__(self, logger: logging.Logger, total: int, desc: str = "处理中"):
        self.logger = logger
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
        
    def update(self, n: int = 1) -> None:
        """更新进度"""
        self.current += n
        if self.current % max(1, self.total // 10) == 0 or self.current == self.total:
            self._log_progress()
            
    def _log_progress(self) -> None:
        """输出进度日志"""
        elapsed = time.time() - self.start_time
        percent = self.current / self.total * 100
        rate = self.current / elapsed if elapsed > 0 else 0
        
        self.logger.info(
            "%s: %d/%d (%.1f%%) | 速率: %.1f items/s | 已耗时: %.1fs",
            self.desc, self.current, self.total, percent, rate, elapsed
        )
        
    def complete(self) -> None:
        """完成日志"""
        elapsed = time.time() - self.start_time
        self.logger.info(
            "%s 完成: %d 项 | 总耗时: %.3fs | 平均: %.3f s/item",
            self.desc, self.current, elapsed, elapsed / self.current if self.current > 0 else 0
        )
