"""test_config.py - 主动回复 v3 配置测试"""

from __future__ import annotations

import pytest

from iris_memory.proactive.config import (
    FollowUpConfig,
    ProactiveConfig,
    SignalQueueConfig,
)


class TestSignalQueueConfig:
    """SignalQueueConfig 默认值测试"""

    def test_defaults(self) -> None:
        c = SignalQueueConfig()
        assert c.check_interval_seconds == 30
        assert c.silence_timeout_seconds == 600
        assert c.min_silence_seconds == 60
        assert c.ttl_emotion_high == 180
        assert c.ttl_rule_match == 300
        assert c.weight_direct_reply == 0.6
        assert c.weight_llm_confirm == 0.4
        assert c.max_signals_per_group == 50

    def test_override(self) -> None:
        c = SignalQueueConfig(check_interval_seconds=10, max_signals_per_group=20)
        assert c.check_interval_seconds == 10
        assert c.max_signals_per_group == 20


class TestFollowUpConfig:
    """FollowUpConfig 默认值测试"""

    def test_defaults(self) -> None:
        c = FollowUpConfig()
        assert c.short_window_seconds == 10
        assert c.llm_max_tokens == 500
        assert c.llm_temperature == 0.3
        assert c.fallback_to_rule_on_llm_error is True

    def test_override(self) -> None:
        c = FollowUpConfig(fallback_to_rule_on_llm_error=False)
        assert c.fallback_to_rule_on_llm_error is False


class TestProactiveConfig:
    """ProactiveConfig 完整配置测试"""

    def test_defaults(self) -> None:
        c = ProactiveConfig()
        assert c.enabled is False
        assert c.signal_queue_enabled is True
        assert c.followup_enabled is True
        assert c.followup_window_seconds == 120
        assert c.max_followup_count == 2
        assert c.max_reply_tokens == 150
        assert c.reply_temperature == 0.7
        assert c.cooldown_seconds == 60
        assert c.max_daily_replies == 20
        assert c.max_daily_per_user == 5
        assert c.group_whitelist_mode is False
        assert c.group_whitelist == []
        assert c.quiet_hours == [23, 7]
        assert c.proactive_mode == "rule"
        assert isinstance(c.signal_queue, SignalQueueConfig)
        assert isinstance(c.followup, FollowUpConfig)

    def test_override_nested(self) -> None:
        sq = SignalQueueConfig(check_interval_seconds=15)
        fu = FollowUpConfig(short_window_seconds=20)
        c = ProactiveConfig(
            enabled=True,
            proactive_mode="hybrid",
            signal_queue=sq,
            followup=fu,
        )
        assert c.enabled is True
        assert c.proactive_mode == "hybrid"
        assert c.signal_queue.check_interval_seconds == 15
        assert c.followup.short_window_seconds == 20

    def test_whitelist_config(self) -> None:
        c = ProactiveConfig(
            group_whitelist_mode=True,
            group_whitelist=["g1", "g2"],
        )
        assert c.group_whitelist_mode is True
        assert c.group_whitelist == ["g1", "g2"]

    def test_quiet_hours_custom(self) -> None:
        c = ProactiveConfig(quiet_hours=[1, 6])
        assert c.quiet_hours == [1, 6]
