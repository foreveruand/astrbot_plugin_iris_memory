"""test_signal_queue.py - 信号队列测试"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from iris_memory.proactive.config import ProactiveConfig, SignalQueueConfig
from iris_memory.proactive.models import Signal, SignalType
from iris_memory.proactive.signal_queue import SignalQueue


def _make_signal(
    group_id: str = "g1",
    user_id: str = "u1",
    weight: float = 0.5,
    signal_type: SignalType = SignalType.RULE_MATCH,
    expires_at=None,
) -> Signal:
    return Signal(
        signal_type=signal_type,
        session_key=f"{user_id}:{group_id}",
        group_id=group_id,
        user_id=user_id,
        weight=weight,
        expires_at=expires_at,
    )


@pytest.fixture
def config() -> ProactiveConfig:
    return ProactiveConfig(
        signal_queue=SignalQueueConfig(max_signals_per_group=5)
    )


@pytest.fixture
def queue(config: ProactiveConfig) -> SignalQueue:
    return SignalQueue(config)


class TestEnqueue:
    """信号入队测试"""

    def test_enqueue_single(self, queue: SignalQueue) -> None:
        s = _make_signal()
        assert queue.enqueue(s) is True
        assert queue.total_signals == 1

    def test_enqueue_multiple_groups(self, queue: SignalQueue) -> None:
        queue.enqueue(_make_signal(group_id="g1"))
        queue.enqueue(_make_signal(group_id="g2"))
        assert queue.group_count == 2
        assert queue.total_signals == 2

    def test_enqueue_overflow_removes_oldest(self, queue: SignalQueue) -> None:
        """超出容量限制时移除最早的信号"""
        for i in range(6):
            queue.enqueue(_make_signal(weight=0.1 * i))
        # max=5, 入了6个，最早的被移除
        signals = queue.get_signals("g1")
        assert len(signals) == 5
        # 最早的 weight=0.0 应该被移除
        assert all(s.weight >= 0.1 for s in signals)


class TestGetSignals:
    """获取信号测试"""

    def test_get_empty(self, queue: SignalQueue) -> None:
        assert queue.get_signals("nonexistent") == []

    def test_get_filters_expired(self, queue: SignalQueue) -> None:
        # 一个未过期，一个已过期
        queue.enqueue(_make_signal(weight=0.3))
        queue.enqueue(_make_signal(
            weight=0.7,
            expires_at=datetime.now() - timedelta(seconds=1),
        ))
        signals = queue.get_signals("g1")
        assert len(signals) == 1
        assert signals[0].weight == 0.3

    def test_get_keeps_valid(self, queue: SignalQueue) -> None:
        queue.enqueue(_make_signal(
            weight=0.5,
            expires_at=datetime.now() + timedelta(hours=1),
        ))
        signals = queue.get_signals("g1")
        assert len(signals) == 1

    def test_get_removes_all_expired(self, queue: SignalQueue) -> None:
        """所有信号都过期时，队列应被清空"""
        queue.enqueue(_make_signal(
            expires_at=datetime.now() - timedelta(seconds=10),
        ))
        signals = queue.get_signals("g1")
        assert len(signals) == 0


class TestClearSession:
    """清除会话信号测试"""

    def test_clear_specific_session(self, queue: SignalQueue) -> None:
        queue.enqueue(_make_signal(user_id="u1", group_id="g1"))
        queue.enqueue(_make_signal(user_id="u2", group_id="g1"))
        removed = queue.clear_session("u1:g1")
        assert removed == 1
        signals = queue.get_signals("g1")
        assert len(signals) == 1
        assert signals[0].user_id == "u2"

    def test_clear_nonexistent_session(self, queue: SignalQueue) -> None:
        removed = queue.clear_session("nobody:g1")
        assert removed == 0

    def test_clear_removes_empty_group(self, queue: SignalQueue) -> None:
        queue.enqueue(_make_signal(user_id="u1", group_id="g1"))
        queue.clear_session("u1:g1")
        assert queue.group_count == 0


class TestClearGroup:
    """清除群组信号测试"""

    def test_clear_group(self, queue: SignalQueue) -> None:
        queue.enqueue(_make_signal(group_id="g1"))
        queue.enqueue(_make_signal(group_id="g1", user_id="u2"))
        removed = queue.clear_group("g1")
        assert removed == 2
        assert queue.total_signals == 0

    def test_clear_nonexistent_group(self, queue: SignalQueue) -> None:
        assert queue.clear_group("no_group") == 0


class TestLastMessageTime:
    """最后消息时间跟踪测试"""

    def test_update_and_get(self, queue: SignalQueue) -> None:
        queue.update_last_message_time("g1")
        t = queue.get_last_message_time("g1")
        assert t is not None
        assert (datetime.now() - t).total_seconds() < 1

    def test_get_nonexistent(self, queue: SignalQueue) -> None:
        assert queue.get_last_message_time("nope") is None

    def test_silence_duration_no_record(self, queue: SignalQueue) -> None:
        assert queue.get_silence_duration("nope") == float("inf")

    def test_silence_duration_after_update(self, queue: SignalQueue) -> None:
        queue.update_last_message_time("g1")
        d = queue.get_silence_duration("g1")
        assert 0 <= d < 1


class TestAggregateWeight:
    """聚合权重计算测试"""

    def test_empty_group(self, queue: SignalQueue) -> None:
        assert queue.aggregate_weight("g1") == 0.0

    def test_single_signal(self, queue: SignalQueue) -> None:
        queue.enqueue(_make_signal(weight=0.6))
        assert queue.aggregate_weight("g1") == pytest.approx(0.6)

    def test_multiple_signals_max_plus_decay(self, queue: SignalQueue) -> None:
        """最大值 + 其余 0.3 衰减叠加"""
        queue.enqueue(_make_signal(weight=0.6))
        queue.enqueue(_make_signal(weight=0.4, user_id="u2"))
        # base=0.6, bonus=0.4*0.3=0.12 → 0.72
        assert queue.aggregate_weight("g1") == pytest.approx(0.72)

    def test_capped_at_1(self, queue: SignalQueue) -> None:
        """总权重不超过 1.0"""
        queue.enqueue(_make_signal(weight=0.9))
        queue.enqueue(_make_signal(weight=0.8, user_id="u2"))
        queue.enqueue(_make_signal(weight=0.7, user_id="u3"))
        # base=0.9, bonus=(0.8+0.7)*0.3=0.45 → 1.35 → capped at 1.0
        assert queue.aggregate_weight("g1") == 1.0


class TestActiveGroups:
    """活跃群组列表测试"""

    def test_no_groups(self, queue: SignalQueue) -> None:
        assert queue.get_active_groups() == []

    def test_multiple_groups(self, queue: SignalQueue) -> None:
        queue.enqueue(_make_signal(group_id="g1"))
        queue.enqueue(_make_signal(group_id="g2"))
        groups = queue.get_active_groups()
        assert set(groups) == {"g1", "g2"}
