from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillDefinition:
    """Skill 配置定义。"""

    skill_id: str
    name: str
    channel: str
    strategy: str
    description: str = ""
    tool_scene_keywords: list[str] = field(default_factory=list)
    hide_sources: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillDefinition":
        return cls(
            skill_id=str(payload.get("id", "")).strip(),
            name=str(payload.get("name", "")).strip(),
            channel=str(payload.get("channel", "")).strip().lower(),
            strategy=str(payload.get("strategy", "")).strip(),
            description=str(payload.get("description", "")).strip(),
            tool_scene_keywords=[
                str(x).strip().lower()
                for x in (payload.get("tool_scene_keywords") or [])
                if str(x).strip()
            ],
            hide_sources=bool(payload.get("response", {}).get("hide_sources", True)),
        )
