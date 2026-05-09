# 工具脚本

本目录包含项目开发和测试使用的各种辅助脚本。

## 📁 目录结构

```
scripts/
├── dev/      # 开发工具
└── test/     # 测试脚本
```

## 🛠️ 开发工具 (dev/)

用于日常开发的辅助工具：

### run.py
快速启动开发服务器。

```bash
python scripts/dev/run.py
```

功能：
- 自动加载环境变量
- 热重载支持
- 显示启动信息

### verify_compression.sh
验证压缩功能的 shell 脚本。

```bash
bash scripts/dev/verify_compression.sh
```

功能：
- 测试所有压缩策略
- 对比压缩效果
- 生成测试报告

## 🧪 测试脚本 (test/)

专门用于测试的脚本：

### test_langfuse_fixed.py
验证 Langfuse 集成修复效果。

```bash
python scripts/test/test_langfuse_fixed.py
```

功能：
- 测试 Langfuse 基本功能
- 验证错误修复
- 检查追踪数据

### test_langfuse_debug.py
Langfuse 调试工具。

```bash
python scripts/test/test_langfuse_debug.py
```

功能：
- 详细的调试输出
- 错误追踪
- 性能分析

## 🚀 使用建议

### 开发阶段

使用 `run.py` 快速启动服务：

```bash
# 启动开发服务器
python scripts/dev/run.py

# 或者直接运行 main.py
python app/main.py
```

### 功能验证

使用测试脚本验证新功能：

```bash
# 验证压缩功能
bash scripts/dev/verify_compression.sh

# 验证 Langfuse
python scripts/test/test_langfuse_fixed.py
```

### 持续集成

在 CI/CD 流程中使用：

```bash
# 运行所有测试脚本
python scripts/test/test_langfuse_fixed.py
python scripts/test/test_langfuse_debug.py
```

## 📝 添加新脚本

### 开发工具

如果你创建了新的开发工具，放在 `dev/` 目录：

```bash
# 新建脚本
touch scripts/dev/my_tool.py

# 添加执行权限（如果需要）
chmod +x scripts/dev/my_tool.py
```

### 测试脚本

如果你创建了新的测试脚本，放在 `test/` 目录：

```bash
# 新建脚本
touch scripts/test/test_my_feature.py
```

### 脚本规范

1. 使用清晰的文件名
2. 添加顶部注释说明用途
3. 提供使用示例
4. 处理错误情况
5. 输出有意义的信息

示例：

```python
#!/usr/bin/env python3
"""
测试我的功能

用法:
    python scripts/test/test_my_feature.py

说明:
    这个脚本用于...
"""

import sys

def main():
    try:
        # 你的代码
        print("✅ 测试通过")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## 🔍 常见问题

**Q: 脚本找不到模块？**

A: 确保在项目根目录运行脚本，或者添加路径：

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
```

**Q: 脚本权限不足？**

A: 添加执行权限：

```bash
chmod +x scripts/dev/your_script.sh
```

**Q: 如何在 Windows 上运行 .sh 脚本？**

A: 使用 Git Bash 或 WSL，或者将脚本改写为 Python 版本。
