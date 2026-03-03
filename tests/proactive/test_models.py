"""test_models.py - 主动回复 v3 数据模型测试"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from iris_memory.proactive.models import (
    AggregatedDecision,
    FollowUpDecision,
    FollowUpExpectation,
    FollowUpReplyType,
    ProactiveReplyResult,
    Signal,
    SignalType,
)


class TestSignalType:
    """SignalType 枚举测试"""

    def test_values(self) -> None:
        assert SignalType.EMOTION_HIGH.value == "emotion_high"
        assert SignalType.RULE_MATCH.value == "rule_match"

    def test_is_str_enum(self) -> None:
        assert isinstance(SignalType.EMOTION_HIGH, str)


class TestFollowUpReplyType:
    """FollowUpReplyType 枚举测试"""

    def test_values(self) -> None:
        assert FollowUpReplyType.ACKNOWLEDGE.value == "acknowledge"
        assert FollowUpReplyType.CONTINUE_TOPIC.value == "continue_topic"
        assert FollowUpReplyType.EMOTION_SUPPORT.value == "emotion_support"
        assert FollowUpReplyType.QUESTION.value == "question"


class TestSignal:
    """Signal 数据模型测试"""

    def test_create_basic_signal(self) -> None:
        s = Signal(
            signal_type=SignalType.RULE_MATCH,
            session_key="user1:group1",
            group_id="group1",
            user_id="user1",
            weight=0.5,
        )
        assert s.signal_type == SignalType.RULE_MATCH
        assert s.session_key == "user1:group1"
        assert s.group_id == "group1"
        assert s.user_id == "user1"
        assert s.weight == 0.5
        assert s.signal_id  # 自动生成
        assert isinstance(s.created_at, datetime)
        assert s.expires_at is None
        assert s.metadata == {}

    def test_signal_not_expired_when_no_expiry(self) -> None:
        s = Signal(
            signal_type=SignalType.EMOTION_HIGH,
            session_key="u:g",
            group_id="g",
            user_id="u",
            weight=0.8,
        )
        assert not s.is_expired

    def test_signal_not_expired_when_future(self) -> None:
        s = Signal(
            signal_type=SignalType.RULE_MATCH,
            session_key="u:g",
            group_id="g",
            user_id="u",
            weight=0.3,
            expires_at=datetime.now() + timedelta(hours=1),
        )
        assert not s.is_expired

    def test_signal_expired_when_past(self) -> None:
        s = Signal(
            signal_type=SignalType.RULE_MATCH,
            session_key="u:g",
            group_id="g",
            user_id="u",
            weight=0.3,
            expires_at=datetime.now() - timedelta(seconds=1),
        )
        assert s.is_expired

    def test_signal_metadata(self) -> None:
        s = Signal(
            signal_type=SignalType.RULE_MATCH,
            session_key="u:g",
            group_id="g",
            user_id="u",
            weight=0.4,
            metadata={"matched_rules": ["question"], "text_preview": "hello"},
        )
        assert s.metadata["matched_rules"] == ["question"]

    def test_signal_id_unique(self) -> None:
        s1 = Signal(signal_type=SignalType.RULE_MATCH, session_key="u:g", group_id="g", user_id="u", weight=0.5)
        s2 = Signal(signal_type=SignalType.RULE_MATCH, session_key="u:g", group_id="g", user_id="u", weight=0.5)
        assert s1.signal_id != s2.signal_id


class TestFollowUpExpectation:
    """FollowUpExpectation 数据模型测试"""

    def _make_expectation(self, **kwargs) -> FollowUpExpectation:
        defaults = dict(
            session_key="user1:group1",
            group_id="group1",
            trigger_user_id="user1",
            trigger_message="你好",
            bot_reply_summary="回复了你好",
            followup_window_end=datetime.now() + timedelta(minutes=2),
        )
        defaults.update(kwargs)
        return FollowUpExpectation(**defaults)

    def test_create_basic(self) -> None:
        e = self._make_expectation()
        assert e.group_id == "group1"
        assert e.trigger_user_id == "user1"
        assert e.followup_count == 0
        assert not e.is_window_expired
        assert not e.has_aggregated_messages

    def test_window_expired(self) -> None:
        e = self._make_expectation(
            followup_window_end=datetime.now() - timedelta(seconds=1)
        )
        assert e.is_window_expired

    def test_short_window_not_expired_when_none(self) -> None:
        e = self._make_expectation()
        assert not e.is_short_window_expired

    def test_short_window_expired(self) -> None:
        e = self._make_expectation(
            short_window_end=datetime.now() - timedelta(seconds=1)
        )
        assert e.is_short_window_expired

    def test_short_window_not_expired_when_future(self) -> None:
        e = self._make_expectation(
            short_window_end=datetime.now() + timedelta(seconds=10)
        )
        assert not e.is_short_window_expired

    def test_has_aggregated_messages(self) -> None:
        e = self._make_expectation()
        assert not e.has_aggregated_messages
        e.aggregated_messages.append({"content": "test"})
        assert e.has_aggregated_messages


class TestAggregatedDecision:
    """AggregatedDecision 数据模型测试"""

    def test_defaults(self) -> None:
        d = AggregatedDecision(
            should_reply=True,
            session_key="u:g",
            group_id="g",
        )
        assert d.should_reply is True
        assert d.target_user_id == ""
        assert d.aggregated_weight == 0.0
        assert d.signals == []
        assert d.reason == ""
        assert d.recent_messages == []
        assert d.llm_confirmed is False


class TestFollowUpDecision:
    """FollowUpDecision 数据模型测试"""

    def test_defaults(self) -> None:
        d = FollowUpDecision(should_reply=False)
        assert d.reason == ""
        assert d.reply_type == FollowUpReplyType.ACKNOWLEDGE
        assert d.suggested_direction == ""


class TestProactiveReplyResult:
    """ProactiveReplyResult 数据模型测试"""

    def test_defaults(self) -> None:
        r = ProactiveReplyResult(trigger_prompt="test prompt")
        assert r.trigger_prompt == "test prompt"
        assert r.reply_params == {}
        assert r.reason == ""
        assert r.source == "signal_queue"

    def test_followup_source(self) -> None:
        r = ProactiveReplyResult(trigger_prompt="p", source="followup")
        assert r.source == "followup"
