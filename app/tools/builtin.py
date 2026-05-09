"""
内置工具集合
包含项目预定义的常用工具
"""

from langchain.tools import tool
from datetime import datetime
from typing import Optional
import ast
import operator


@tool
def get_current_date_tool(format: Optional[str] = None) -> str:
    """获取当前日期和时间"""
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
    """获取当前 Unix 时间戳"""
    return str(int(datetime.now().timestamp()))


@tool
def get_weekday() -> str:
    """获取今天是星期几"""
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


# ========================================
# 新增工具
# ========================================

@tool
def calculator(expression: str) -> str:
    """安全的数学计算器，支持基本运算"""
    # 定义允许的操作符（安全）
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
    }

    def safe_eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.BinOp):
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            return allowed_operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = safe_eval(node.operand)
            return allowed_operators[type(node.op)](operand)
        else:
            raise ValueError(f"不支持的操作: {ast.dump(node)}")

    try:
        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree.body)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "错误: 除数不能为零"
    except KeyError as e:
        return f"错误: 不支持的运算符"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """使用 DuckDuckGo 进行网络搜索"""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "未找到相关结果"

        output = f"搜索 '{query}' 的结果:\n\n"
        for i, result in enumerate(results, 1):
            output += f"{i}. {result['title']}\n"
            output += f"   {result['body'][:200]}{'...' if len(result['body']) > 200 else ''}\n"
            output += f"   链接: {result['href']}\n\n"

        return output
    except ImportError:
        return "错误: 需要安装 duckduckgo-search 库 (pip install duckduckgo-search)"
    except Exception as e:
        return f"搜索失败: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    try:
        import requests

        # 使用免费天气API wttr.in
        url = f"https://wttr.in/{city}?format=j1&lang=zh"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        current = data['current_condition'][0]
        weather_desc = current.get('lang_zh', [{}])[0].get('value', current.get('weatherDesc', [{}])[0].get('value', '未知'))

        return f"""{city}当前天气:
- 温度: {current['temp_C']}°C (体感 {current['FeelsLikeC']}°C)
- 天气: {weather_desc}
- 湿度: {current['humidity']}%
- 风速: {current['windspeedKmph']} km/h
- 能见度: {current['visibility']} km"""
    except ImportError:
        return "错误: 需要安装 requests 库 (pip install requests)"
    except requests.exceptions.Timeout:
        return "错误: 请求超时，请稍后重试"
    except requests.exceptions.RequestException as e:
        return f"获取天气失败: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"解析天气数据失败: 城市名称可能不正确"
    except Exception as e:
        return f"获取天气失败: {str(e)}"


@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """货币汇率转换"""
    try:
        import requests

        # 使用免费汇率API
        url = f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if to_currency.upper() not in data['rates']:
            return f"错误: 不支持的货币代码 '{to_currency}'"

        rate = data['rates'][to_currency.upper()]
        result = amount * rate

        return f"{amount} {from_currency.upper()} = {result:.2f} {to_currency.upper()}\n(汇率: 1 {from_currency.upper()} = {rate:.4f} {to_currency.upper()})"
    except ImportError:
        return "错误: 需要安装 requests 库 (pip install requests)"
    except requests.exceptions.Timeout:
        return "错误: 请求超时，请稍后重试"
    except requests.exceptions.RequestException as e:
        return f"汇率转换失败: 网络错误"
    except KeyError:
        return f"错误: 不支持的货币代码 '{from_currency}'"
    except Exception as e:
        return f"汇率转换失败: {str(e)}"


@tool
def fetch_url_content(url: str, max_length: int = 2000) -> str:
    """抓取网页内容"""
    try:
        import requests
        from bs4 import BeautifulSoup

        # 发送请求
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()

        # 解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # 移除script和style标签
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # 提取文本
        text = soup.get_text(separator='\n', strip=True)

        # 清理多余空行
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)

        # 限制长度
        if len(text) > max_length:
            text = text[:max_length] + "...\n(内容过长，已截断)"

        return f"网页标题: {soup.title.string if soup.title else '无标题'}\n网页URL: {url}\n\n内容:\n{text}"
    except ImportError:
        return "错误: 需要安装 requests 和 beautifulsoup4 库 (pip install requests beautifulsoup4)"
    except requests.exceptions.Timeout:
        return "错误: 请求超时，网页响应过慢"
    except requests.exceptions.RequestException as e:
        return f"抓取失败: 无法访问该网页"
    except Exception as e:
        return f"抓取失败: {str(e)}"


@tool
def python_executor(code: str) -> str:
    """安全的 Python 代码执行器"""
    import sys
    from io import StringIO
    import contextlib

    # 限制可用的内置函数（安全）
    safe_builtins = {
        'abs': abs, 'all': all, 'any': any, 'bool': bool,
        'dict': dict, 'enumerate': enumerate, 'float': float,
        'int': int, 'len': len, 'list': list, 'max': max,
        'min': min, 'pow': pow, 'range': range, 'round': round,
        'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple,
        'zip': zip, 'print': print, 'map': map, 'filter': filter,
        'set': set, 'frozenset': frozenset, 'chr': chr, 'ord': ord,
    }

    # 检查危险关键字
    dangerous_keywords = ['import', 'open', 'eval', 'exec', '__', 'compile', 'globals', 'locals']
    for keyword in dangerous_keywords:
        if keyword in code:
            return f"安全错误: 代码中包含禁用的关键字 '{keyword}'"

    # 捕获输出
    output = StringIO()

    try:
        with contextlib.redirect_stdout(output):
            # 执行代码
            exec(code, {"__builtins__": safe_builtins}, {})

        result = output.getvalue()
        return result if result else "代码执行成功（无输出）"
    except Exception as e:
        return f"执行错误: {type(e).__name__}: {str(e)}"


# 导出所有内置工具
BUILTIN_TOOLS = [
    get_current_date_tool,
    get_current_timestamp,
    get_weekday,
    calculator,
    web_search,
    get_weather,
    currency_converter,
    fetch_url_content,
    python_executor,
]
