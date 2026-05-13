# Skills 目录规范

## 目录划分

- `app/skills/defs/`
  - 存放通用 skill（可提交到 Git）
  - 示例：`feishu_rag_first.json`

- `app/skills/sensitive/`
  - 存放敏感 skill（不会提交到 Git）
  - 适合放含内部策略、私有规则、临时实验配置
  - 示例：`resp_style.json`

## 加载优先级

SkillLoader 按以下优先级加载：
1. 优先从 `sensitive/` 目录加载
2. 如果找不到，回退到 `defs/` 目录

这样可以：
- 在 `sensitive/` 中覆盖 `defs/` 中的同名 skill
- 保持敏感配置不进入版本控制
- 维持通用 skill 的版本管理

## 命名约定

- 通用 skill：`<skill_id>.json`
- 敏感 skill：`<skill_id>.json`（放在 `sensitive/` 目录即可）

## 当前忽略规则

已在仓库根目录 `.gitignore` 中配置：

- `app/skills/sensitive/`（整个目录）
- `app/skills/**/*.sensitive.json`（任意位置的 .sensitive.json 文件）

这意味着：

- 通用 skill 默认可追踪
- 敏感 skill 默认不会进入版本库
- `sensitive/` 目录中的所有文件都不会被提交

## Skill 配置示例

### 通用 Skill (defs/feishu_rag_first.json)
```json
{
  "id": "feishu_rag_first",
  "name": "Feishu RAG First",
  "channel": "feishu",
  "strategy": "rag_first_then_tool_summary",
  "description": "飞书问答优先走 RAG，再按场景补充工具",
  "tool_scene_keywords": ["实时", "最新", "天气"],
  "response": {
    "hide_sources": true
  }
}
```

### 敏感 Skill (sensitive/resp_style.json)
```json
{
  "id": "resp_style",
  "name": "Response Style",
  "channel": "feishu",
  "strategy": "default",
  "style_instruction": "你的风格指令...",
  "response": {
    "hide_sources": true
  }
}
```

## 使用方式

在 `.env` 中配置：
```bash
FEISHU_SKILL_ID=resp_style
```

SkillLoader 会自动按优先级查找并加载。
