# Skills 目录规范

## 目录划分

- `app/skills/defs/`
  - 存放通用 skill（可提交到 Git）
  - 示例：`feishu_rag_first.json`

- `app/skills/sensitive/`
  - 存放敏感 skill（不会提交到 Git）
  - 适合放含内部策略、私有规则、临时实验配置

## 命名约定

- 通用 skill：`<skill_id>.json`
- 敏感 skill：`<skill_id>.sensitive.json`（同样会被忽略）

## 当前忽略规则

已在仓库根目录 `.gitignore` 中配置：

- `app/skills/sensitive/`
- `app/skills/**/*.sensitive.json`

这意味着：

- 通用 skill 默认可追踪
- 敏感 skill 默认不会进入版本库
