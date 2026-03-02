"""
群冷却管理器

核心逻辑：管理群聊冷却状态的激活、取消、查询。
使用 BoundedDict 存储，重启自动重置。
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Final

from iris_memory.cooldown.cooldown_state import CooldownState
from iris_memory.utils.bounded_dict import BoundedDict
from iris_memory.utils.logger import get_logger

logger = get_logger("cooldown_manager")

# ── 硬编码默认值 ──

DEFAULT_DURATION_MINUTES: Final[int] = 20
"""默认冷却时长（分钟）"""

MIN_DURATION_MINUTES: Final[int] = 5
"""最小冷却时长（分钟）"""

MAX_DURATION_MINUTES: Final[int] = 180
"""最大冷却时长（分钟，3小时）"""

MAX_TRACKED_GROUPS: Final[int] = 1000
"""最大跟踪群数"""

# ── 时长解析正则 ──

_DURATION_PATTERN: Final = re.compile(
    r"^(\d+)\s*(m|min|分钟?|h|hour|小时)?$",
    re.IGNORECASE,
)


def parse_duration(raw: str) -> Optional[int]:
    """解析时长字符串为分钟数

    支持格式:
    - 纯数字: 视为分钟（如 "30"）
    - 数字+单位: 30m, 1h, 30分钟, 1小时

    Args:
        raw: 原始时长字符串

    Returns:
        解析后的分钟数，解析失败返回 None
    """
    raw = raw.strip()
    if not raw:
        return None

    match = _DURATION_PATTERN.match(raw)
    if not match:
        return None

    value = int(match.group(1))
    unit = (match.group(2) or "m").lower()

    if unit in ("h", "hour", "小时"):
        return value * 60
    return value


class CooldownManager:
    """群冷却管理器

    管理群聊冷却状态，提供激活、取消、查询接口。
    冷却状态存储在内存中（BoundedDict），插件重启自动重置。

    线程安全：所有公共方法均为同步操作，通过 BoundedDict 保证容量限制。
    """

    def __init__(
        self,
        default_duration: int = DEFAULT_DURATION_MINUTES,
        max_groups: int = MAX_TRACKED_GROUPS,
    ) -> None:
        """
        初始化冷却管理器

        Args:
            default_duration: 默认冷却时长（分钟）
            max_groups: 最大跟踪群数
        """
        self._default_duration = default_duration
        self._states: BoundedDict[str, CooldownState] = BoundedDict(max_size=max_groups)

    @property
    def default_duration(self) -> int:
        """默认冷却时长（分钟）"""
        return self._default_duration

    def activate(
        self,
        group_id: str,
        duration_minutes: Optional[int] = None,
        reason: Optional[str] = None,
        initiated_by: str = "user",
    ) -> str:
        """激活群冷却

        Args:
            group_id: 群聊 ID
            duration_minutes: 冷却时长（分钟），None 使用默认值
            reason: 冷却原因说明
            initiated_by: 触发者类型，"user" 或 "llm"

        Returns:
            str: 操作结果消息
        """
        duration = duration_minutes or self._default_duration

        # 范围校验
        if duration < MIN_DURATION_MINUTES:
            return f"冷却时间不能少于 {MIN_DURATION_MINUTES} 分钟"
        if duration > MAX_DURATION_MINUTES:
            return f"冷却时间不能超过 {MAX_DURATION_MINUTES} 分钟（{MAX_DURATION_MINUTES // 60} 小时）"

        now = datetime.now(timezone.utc)
        state = CooldownState(
            group_id=group_id,
            started_at=now,
            expires_at=now + timedelta(minutes=duration),
            initiated_by=initiated_by,
            reason=reason,
        )
        self._states[group_id] = state

        logger.info(
            f"Cooldown activated: group={group_id}, duration={duration}min, "
            f"initiated_by={initiated_by}, reason={reason}"
        )

        reason_line = f"\n原因：{reason}" if reason else ""
        return (
            f"⏸️ 已进入冷却模式（{duration}分钟）{reason_line}\n"
            f"期间我将暂停主动回复，仅响应@消息和指令"
        )

    def deactivate(self, group_id: str) -> str:
        """取消群冷却

        Args:
            group_id: 群聊 ID

        Returns:
            str: 操作结果消息
        """
        if group_id in self._states:
            del self._states[group_id]
            logger.info(f"Cooldown deactivated: group={group_id}")
            return "▶️ 已退出冷却模式，恢复正常服务"

        return "当前群聊未处于冷却模式"

    def get_status(self, group_id: str) -> Optional[CooldownState]:
        """获取群冷却状态

        如果冷却已过期，自动清理并返回 None。

        Args:
            group_id: 群聊 ID

        Returns:
            CooldownState 或 None（无冷却或已过期）
        """
        state = self._states.get(group_id)
        if state is None:
            return None

        if not state.is_active:
            # 过期自动清理
            del self._states[group_id]
            logger.debug(f"Cooldown expired and cleaned: group={group_id}")
            return None

        return state

    def is_active(self, group_id: str) -> bool:
        """检查群是否处于冷却中

        Args:
            group_id: 群聊 ID

        Returns:
            bool: 是否处于冷却中
        """
        return self.get_status(group_id) is not None

    def format_status(self, group_id: str) -> str:
        """格式化群冷却状态为用户可读字符串

        Args:
            group_id: 群聊 ID

        Returns:
            str: 状态描述
        """
        state = self.get_status(group_id)
        if state is None:
            return "当前群聊未处于冷却模式"

        return (
            f"⏸️ 当前处于冷却模式\n"
            f"剩余时间：{state.format_remaining()}\n"
            f"到期时间：{state.format_expires_at_local()}"
        )

    @property
    def active_count(self) -> int:
        """当前活跃冷却群数"""
        # 简单计数，不触发自动清理
        return sum(1 for s in self._states.values() if s.is_active)
