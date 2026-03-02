"""
冷却状态数据模型

不可变数据类，记录某个群聊的冷却状态信息。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class CooldownState:
    """群冷却状态

    Attributes:
        group_id: 群聊 ID
        started_at: 冷却开始时间（UTC）
        expires_at: 冷却到期时间（UTC）
        initiated_by: 触发者类型，"user" 或 "llm"
        reason: 冷却原因说明
    """
    group_id: str
    started_at: datetime
    expires_at: datetime
    initiated_by: str
    reason: Optional[str] = None

    @property
    def is_active(self) -> bool:
        """冷却是否仍处于生效状态"""
        return datetime.now(timezone.utc) < self.expires_at

    @property
    def remaining_seconds(self) -> int:
        """剩余冷却时间（秒），已过期返回 0"""
        delta = self.expires_at - datetime.now(timezone.utc)
        return max(0, int(delta.total_seconds()))

    @property
    def remaining_minutes(self) -> int:
        """剩余冷却时间（分钟，向上取整），已过期返回 0"""
        return math.ceil(self.remaining_seconds / 60)

    @property
    def duration_minutes(self) -> int:
        """原始冷却时长（分钟）"""
        delta = self.expires_at - self.started_at
        return int(delta.total_seconds() / 60)

    def format_remaining(self) -> str:
        """格式化剩余时间为 '分:秒' 字符串"""
        total = self.remaining_seconds
        minutes, seconds = divmod(total, 60)
        return f"{minutes}分{seconds:02d}秒"

    def format_expires_at_local(self) -> str:
        """格式化到期时间为本地 HH:MM 字符串"""
        local_time = self.expires_at.astimezone()
        return local_time.strftime("%H:%M")
