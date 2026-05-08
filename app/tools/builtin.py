"""
内置工具集合
包含项目预定义的常用工具
"""

from langchain.tools import tool
from datetime import datetime
from typing import Optional


@tool
def get_current_date_tool(format: Optional[str] = None) -> str:
    """获取当前日期和时间

    这个工具可以返回当前的日期和时间信息。

    Args:
        format: 可选的时间格式字符串。
                如果不提供，返回标准格式 "YYYY年MM月DD日 HH:MM:SS"
                可选格式示例：
                - "%Y-%m-%d" -> "2026-05-08"
                - "%Y/%m/%d %H:%M" -> "2026/05/08 14:30"
                - "%Y年%m月%d日" -> "2026年05月08日"

    Returns:
        格式化的日期时间字符串

    Examples:
        >>> get_current_date_tool()
        "2026年05月08日 14:30:45"

        >>> get_current_date_tool("%Y-%m-%d")
        "2026-05-08"
    """
    now = datetime.now()

    # 如果提供了格式，使用指定格式
    if format:
        try:
            return now.strftime(format)
        except Exception as e:
            return f"格式错误: {str(e)}。使用默认格式: {now.strftime('%Y年%m月%d日 %H:%M:%S')}"

    # 默认格式：中文友好
    return now.strftime("%Y年%m月%d日 %H:%M:%S")


@tool
def get_current_timestamp() -> str:
    """获取当前 Unix 时间戳

    返回从 1970-01-01 00:00:00 UTC 到现在的秒数。

    Returns:
        Unix 时间戳字符串（秒）

    Examples:
        >>> get_current_timestamp()
        "1746691845"
    """
    return str(int(datetime.now().timestamp()))


@tool
def get_weekday() -> str:
    """获取今天是星期几

    Returns:
        中文的星期表示，如 "星期一"、"星期二" 等

    Examples:
        >>> get_weekday()
        "星期四"
    """
    weekday_map = {
        0: "星期一",
        1: "星期二",
        2: "星期三",
        3: "星期四",
        4: "星期五",
        5: "星期六",
        6: "星期日",
    }

    weekday = datetime.now().weekday()
    return weekday_map[weekday]


# 导出所有内置工具
BUILTIN_TOOLS = [
    get_current_date_tool,
    get_current_timestamp,
    get_weekday,
]
