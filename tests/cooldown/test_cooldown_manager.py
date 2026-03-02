"""CooldownManager 核心逻辑测试"""
from __future__ import annotations

import pytest

from iris_memory.cooldown.cooldown_manager import (
    CooldownManager,
    parse_duration,
    DEFAULT_DURATION_MINUTES,
    MIN_DURATION_MINUTES,
    MAX_DURATION_MINUTES,
)


class TestParseDuration:
    """时长解析测试"""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("30", 30),
            ("5", 5),
            ("120", 120),
            ("30m", 30),
            ("30min", 30),
            ("30分", 30),
            ("30分钟", 30),
            ("1h", 60),
            ("2h", 120),
            ("1hour", 60),
            ("2小时", 120),
            ("1 h", 60),
        ],
    )
    def test_valid_durations(self, raw: str, expected: int) -> None:
        assert parse_duration(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "abc",
            "1d",
            "1.5h",
            "-10",
            "h",
            "m30",
        ],
    )
    def test_invalid_durations(self, raw: str) -> None:
        assert parse_duration(raw) is None


class TestCooldownManager:
    """CooldownManager 核心逻辑测试"""

    def test_activate_default_duration(self) -> None:
        mgr = CooldownManager()
        result = mgr.activate("g1")
        assert "⏸️" in result
        assert f"{DEFAULT_DURATION_MINUTES}分钟" in result
        assert mgr.is_active("g1") is True

    def test_activate_custom_duration(self) -> None:
        mgr = CooldownManager()
        result = mgr.activate("g1", duration_minutes=30)
        assert "30分钟" in result
        assert mgr.is_active("g1") is True

    def test_activate_with_reason(self) -> None:
        mgr = CooldownManager()
        result = mgr.activate("g1", reason="测试原因")
        assert "测试原因" in result

    def test_activate_llm_initiated(self) -> None:
        mgr = CooldownManager()
        mgr.activate("g1", initiated_by="llm")
        state = mgr.get_status("g1")
        assert state is not None
        assert state.initiated_by == "llm"

    def test_activate_below_min_duration(self) -> None:
        mgr = CooldownManager()
        result = mgr.activate("g1", duration_minutes=1)
        assert f"不能少于 {MIN_DURATION_MINUTES}" in result
        assert mgr.is_active("g1") is False

    def test_activate_above_max_duration(self) -> None:
        mgr = CooldownManager()
        result = mgr.activate("g1", duration_minutes=999)
        assert f"不能超过 {MAX_DURATION_MINUTES}" in result
        assert mgr.is_active("g1") is False

    def test_deactivate_active(self) -> None:
        mgr = CooldownManager()
        mgr.activate("g1")
        result = mgr.deactivate("g1")
        assert "▶️" in result
        assert mgr.is_active("g1") is False

    def test_deactivate_not_active(self) -> None:
        mgr = CooldownManager()
        result = mgr.deactivate("g1")
        assert "未处于冷却模式" in result

    def test_get_status_active(self) -> None:
        mgr = CooldownManager()
        mgr.activate("g1")
        state = mgr.get_status("g1")
        assert state is not None
        assert state.group_id == "g1"
        assert state.is_active is True

    def test_get_status_not_set(self) -> None:
        mgr = CooldownManager()
        assert mgr.get_status("g1") is None

    def test_is_active_false_by_default(self) -> None:
        mgr = CooldownManager()
        assert mgr.is_active("g1") is False

    def test_format_status_active(self) -> None:
        mgr = CooldownManager()
        mgr.activate("g1", duration_minutes=30)
        text = mgr.format_status("g1")
        assert "⏸️" in text
        assert "剩余时间" in text
        assert "到期时间" in text

    def test_format_status_not_active(self) -> None:
        mgr = CooldownManager()
        text = mgr.format_status("g1")
        assert "未处于冷却模式" in text

    def test_overwrite_existing_cooldown(self) -> None:
        mgr = CooldownManager()
        mgr.activate("g1", duration_minutes=10)
        mgr.activate("g1", duration_minutes=60)
        state = mgr.get_status("g1")
        assert state is not None
        assert state.duration_minutes == 60

    def test_multiple_groups(self) -> None:
        mgr = CooldownManager()
        mgr.activate("g1", duration_minutes=20)
        mgr.activate("g2", duration_minutes=30)
        assert mgr.is_active("g1") is True
        assert mgr.is_active("g2") is True
        assert mgr.active_count == 2

    def test_active_count(self) -> None:
        mgr = CooldownManager()
        assert mgr.active_count == 0
        mgr.activate("g1")
        assert mgr.active_count == 1

    def test_bounded_dict_capacity(self) -> None:
        mgr = CooldownManager(max_groups=3)
        mgr.activate("g1")
        mgr.activate("g2")
        mgr.activate("g3")
        mgr.activate("g4")  # 超出容量，g1 应被淘汰
        assert mgr.is_active("g1") is False
        assert mgr.is_active("g4") is True


class TestCooldownModule:
    """CooldownModule Feature Module 测试"""

    def test_init(self) -> None:
        from iris_memory.services.modules.cooldown_module import CooldownModule
        mod = CooldownModule()
        assert mod.cooldown_manager is not None

    def test_is_active_none_group(self) -> None:
        from iris_memory.services.modules.cooldown_module import CooldownModule
        mod = CooldownModule()
        assert mod.is_active(None) is False

    def test_is_active_delegates(self) -> None:
        from iris_memory.services.modules.cooldown_module import CooldownModule
        mod = CooldownModule()
        mod.cooldown_manager.activate("g1")
        assert mod.is_active("g1") is True
        assert mod.is_active("g2") is False
