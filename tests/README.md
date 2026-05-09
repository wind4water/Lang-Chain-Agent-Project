# 测试说明

本目录包含项目的所有测试代码，按类型分类组织。

## 📁 目录结构

```
tests/
├── integration/     # 集成测试
├── unit/           # 单元测试
├── demos/          # 演示脚本
└── __init__.py
```

## 🧪 集成测试 (integration/)

测试完整的功能流程和多个组件的集成：

- **test_tool_calling.py** - 工具调用功能测试
- **test_persistence.py** - 持久化存储测试
- **test_persistence_verify.py** - 重启后数据保留验证
- **test_query_database.py** - 数据库查询接口测试
- **test_new_features.py** - 新功能综合测试
- **test_fix_verification.py** - Bug修复验证测试

运行集成测试：
```bash
# 运行所有集成测试
python -m pytest tests/integration/

# 运行特定测试
python tests/integration/test_tool_calling.py
```

## 🔬 单元测试 (unit/)

测试单个组件或函数的独立功能：

- **test_agent.py** - Agent基础功能测试
- **test_async_sqlite.py** - 异步SQLite功能测试
- **test_quick_verify.py** - 快速验证测试

运行单元测试：
```bash
# 运行所有单元测试
python -m pytest tests/unit/

# 运行特定测试
python tests/unit/test_agent.py
```

## 🎭 演示脚本 (demos/)

演示特定功能的使用示例：

- **demo_session.py** - 会话隔离演示
- **check_storage.py** - 存储位置检查
- **test_compression.py** - 压缩策略对比演示

运行演示：
```bash
# 会话隔离演示
python tests/demos/demo_session.py

# 压缩策略对比
python tests/demos/test_compression.py --strategy all

# 测试超长对话
python tests/demos/test_compression.py --long
```

## 🚀 快速运行所有测试

```bash
# 使用 pytest 运行所有测试
pytest tests/

# 显示详细输出
pytest tests/ -v

# 运行特定目录
pytest tests/integration/
pytest tests/unit/
```

## 📝 编写新测试

### 集成测试

放在 `integration/` 目录，测试多个组件的协作：

```python
# tests/integration/test_my_feature.py
import asyncio
from app.agents.sqlite_with_tools import SqliteAgentWithTools

async def test_my_feature():
    agent = SqliteAgentWithTools()
    response = await agent.chat("测试消息", "test_session")
    assert response is not None
```

### 单元测试

放在 `unit/` 目录，测试单个函数或类：

```python
# tests/unit/test_my_module.py
def test_function():
    result = my_function(input)
    assert result == expected
```

### 演示脚本

放在 `demos/` 目录，提供可运行的示例：

```python
# tests/demos/demo_my_feature.py
if __name__ == "__main__":
    print("演示我的功能...")
    # 演示代码
```

## 🔍 测试覆盖率

查看测试覆盖率：

```bash
# 安装覆盖率工具
pip install pytest-cov

# 运行测试并生成覆盖率报告
pytest tests/ --cov=app --cov-report=html

# 在浏览器中查看报告
open htmlcov/index.html
```

## ⚠️ 注意事项

1. 测试文件名以 `test_` 开头
2. 测试函数名以 `test_` 开头
3. 使用有意义的测试会话ID（如 `test_xxx`）
4. 测试完成后清理数据（调用 `clear_history`）
5. 集成测试可能需要有效的 API 密钥
6. 某些测试可能需要网络连接
