"""CooldownState 数据模型测试"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from iris_memory.cooldown.cooldown_state import CooldownState


class TestCooldownState:
    """CooldownState 基础属性测试"""

    def _make_state(
        self,
        duration_minutes: int = 20,
        started_ago_minutes: int = 0,
        initiated_by: str = "user",
        reason: str | None = None,
    ) -> CooldownState:
        """辅助：创建相对于当前时间的 CooldownState"""
        now = datetime.now(timezone.utc)
        started_at = now - timedelta(minutes=started_ago_minutes)
        expires_at = started_at + timedelta(minutes=duration_minutes)
        return CooldownState(
            group_id="g1",
            started_at=started_at,
            expires_at=expires_at,
            initiated_by=initiated_by,
            reason=reason,
        )

    def test_is_active_when_not_expired(self) -> None:
        state = self._make_state(duration_minutes=30, started_ago_minutes=0)
        assert state.is_active is True

    def test_is_active_when_expired(self) -> None:
        state = self._make_state(duration_minutes=10, started_ago_minutes=15)
        assert state.is_active is False

    def test_remaining_seconds_positive(self) -> None:
        state = self._make_state(duration_minutes=20, started_ago_minutes=5)
        # 大约剩余 15 分钟
        assert 14 * 60 <= state.remaining_seconds <= 15 * 60 + 1

    def test_remaining_seconds_expired(self) -> None:
        state = self._make_state(duration_minutes=5, started_ago_minutes=10)
        assert state.remaining_seconds == 0

    def test_remaining_minutes(self) -> None:
        state = self._make_state(duration_minutes=20, started_ago_minutes=5)
        assert state.remaining_minutes == math.ceil(state.remaining_seconds / 60)

    def test_remaining_minutes_expired(self) -> None:
        state = self._make_state(duration_minutes=5, started_ago_minutes=10)
        assert state.remaining_minutes == 0

    def test_duration_minutes(self) -> None:
        state = self._make_state(duration_minutes=30)
        assert state.duration_minutes == 30

    def test_format_remaining(self) -> None:
        state = self._make_state(duration_minutes=20, started_ago_minutes=5)
        text = state.format_remaining()
        assert "分" in text
        assert "秒" in text

    def test_format_expires_at_local(self) -> None:
        state = self._make_state(duration_minutes=20)
        text = state.format_expires_at_local()
        assert ":" in text  # HH:MM format

    def test_frozen_dataclass(self) -> None:
        state = self._make_state()
        with pytest.raises(AttributeError):
            state.group_id = "other"  # type: ignore[misc]

    def test_reason_optional(self) -> None:
        state_no_reason = self._make_state(reason=None)
        assert state_no_reason.reason is None

        state_with_reason = self._make_state(reason="测试原因")
        assert state_with_reason.reason == "测试原因"
