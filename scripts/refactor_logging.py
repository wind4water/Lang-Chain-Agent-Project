#!/usr/bin/env python3
"""
RAG 日志重构脚本
自动修复常见的日志问题
"""

import re
import sys
from pathlib import Path


def fix_print_statements(content: str) -> str:
    """将 print 语句替换为 logger 调用"""
    # 匹配 print(f"...") 格式
    pattern = r'print\(f"([^"]*)"\)'
    
    def replacer(match):
        msg = match.group(1)
        # 提取变量占位符
        vars_in_msg = re.findall(r'\{(\w+)\}', msg)
        if vars_in_msg:
            # 有变量，使用 % 格式化
            new_msg = re.sub(r'\{(\w+)\}', '%s', msg)
            return f'logger.info("{new_msg}", {", ".join(vars_in_msg)})'
        else:
            return f'logger.info("{msg}")'
    
    return re.sub(pattern, replacer, content)


def fix_fstring_logging(content: str) -> str:
    """将 logger.info(f"...") 改为 logger.info("...", ...)"""
    # 匹配 logger.xxx(f"...") 格式
    pattern = r'(logger\.(?:debug|info|warning|error|critical))\(f"([^"]+)"\)'
    
    def replacer(match):
        method = match.group(1)
        msg = match.group(2)
        
        # 提取变量占位符
        vars_in_msg = re.findall(r'\{([^}:]+)(?::[^}]*)?\}', msg)
        if vars_in_msg:
            # 替换 {var} 为 %s
            new_msg = re.sub(r'\{([^}:]+)(?::[^}]*)?\}', '%s', msg)
            # 转义 %
            new_msg = new_msg.replace('%', '%%')
            new_msg = new_msg.replace('%%s', '%s')
            return f'{method}("{new_msg}", {", ".join(vars_in_msg)})'
        return match.group(0)
    
    return re.sub(pattern, replacer, content)


def add_exception_logging(content: str) -> str:
    """改进异常日志，使用 logger.exception"""
    # 将 logger.error(f"...{e}") 改为 logger.exception("...")
    pattern = r'logger\.error\(f"([^"]*\{e\}[^"]*)"\)'
    return re.sub(pattern, r'logger.exception("\1")', content)


def main():
    if len(sys.argv) < 2:
        print("Usage: python refactor_logging.py <file_or_directory>")
        sys.exit(1)
    
    target = Path(sys.argv[1])
    
    if target.is_file():
        files = [target]
    else:
        files = list(target.rglob("*.py"))
    
    for filepath in files:
        if "__pycache__" in str(filepath):
            continue
            
        print(f"处理: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # 应用修复
        content = fix_print_statements(content)
        content = fix_fstring_logging(content)
        content = add_exception_logging(content)
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ 已更新")
        else:
            print(f"  - 无变更")


if __name__ == "__main__":
    main()
