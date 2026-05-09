# 归档代码

本目录包含项目的旧版本代码和废弃文件，仅供参考。

## ⚠️ 重要说明

**这些文件已废弃，请勿在新项目中使用！**

它们被保留在这里用于：
- 历史记录和版本对比
- 迁移参考
- 了解项目演进过程

## 📦 归档文件

### 旧版 Agent 实现

- **agent.py** - 最早的 Agent 实现（已被 app/agents/ 取代）
- **agent_persistent.py** - 早期持久化版本（已被 sqlite.py 取代）
- **agent_postgres.py** - PostgreSQL 版本原型（已被 app/agents/postgres.py 取代）

### 旧版入口文件

- **main.py** - 旧的服务入口（已被 app/main.py 取代）

## 🆕 当前使用的文件

如果你在寻找最新的实现，请查看：

- **当前 Agent**: `app/agents/sqlite_with_tools.py`
- **服务入口**: `app/main.py`
- **工具系统**: `app/tools/`

## 🗑️ 清理说明

这些文件可以安全删除，但为了保留项目历史，我们暂时保留它们。

如果你确定不再需要这些文件，可以执行：

```bash
# 删除整个 archive 目录
rm -rf archive/
```

## 📚 更多信息

查看项目主文档了解当前架构：
- [README.md](../README.md) - 项目主文档
- [docs/architecture/](../docs/architecture/) - 架构设计文档
