from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .models import SkillDefinition


class SkillLoader:
    """从本地 JSON 文件加载 skill 定义。"""

    def __init__(self, skills_dir: Optional[Path] = None):
        base_dir = Path(__file__).resolve().parent
        self.skills_dir = skills_dir or (base_dir / "defs")
        self._cache: dict[str, SkillDefinition] = {}

    def get(self, skill_id: str) -> Optional[SkillDefinition]:
        key = (skill_id or "").strip()
        if not key:
            return None
        if key in self._cache:
            return self._cache[key]

        file_path = self.skills_dir / f"{key}.json"
        if not file_path.exists():
            return None

        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        skill = SkillDefinition.from_dict(payload)
        if not skill.skill_id:
            return None

        self._cache[key] = skill
        return skill
