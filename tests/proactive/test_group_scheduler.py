"""test_group_scheduler.py - 群定时器调度器测试"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iris_memory.proactive.config import ProactiveConfig, SignalQueueConfig
from iris_memory.proactive.group_scheduler import GroupScheduler
from iris_memory.proactive.models import AggregatedDecision, Signal, SignalType
from iris_memory.proactive.signal_queue import SignalQueue


def _make_signal(
    group_id: str = "g1",
    user_id: str = "u1",
    weight: float = 0.5,
) -> Signal:
    return Signal(
        signal_type=SignalType.RULE_MATCH,
        session_key=f"{user_id}:{group_id}",
        group_id=group_id,
        user_id=user_id,
        weight=weight,
    )


@pytest.fixture
def config() -> ProactiveConfig:
    return ProactiveConfig(
        enabled=True,
        proactive_mode="rule",
        signal_queue=SignalQueueConfig(
            check_interval_seconds=1,       # 快速检查便于测试
            silence_timeout_seconds=5,
            min_silence_seconds=0,           # 立即检查
            weight_direct_reply=0.8,
            weight_llm_confirm=0.5,
        ),
    )


@pytest.fixture
def signal_queue(config: ProactiveConfig) -> SignalQueue:
    return SignalQueue(config)


@pytest.fixture
def reply_callback() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def llm_callback() -> AsyncMock:
    return AsyncMock(return_value=True)


@pytest.fixture
def scheduler(
    config: ProactiveConfig,
    signal_queue: SignalQueue,
    reply_callback: AsyncMock,
) -> GroupScheduler:
    return GroupScheduler(
        config=config,
        signal_queue=signal_queue,
        on_reply=reply_callback,
    )


class TestEnsureTimer:
    """定时器创建测试"""

    @pytest.mark.asyncio
    async def test_creates_timer(self, scheduler: GroupScheduler) -> None:
        scheduler.ensure_timer("g1")
        assert scheduler.has_active_timer("g1")
        assert scheduler.active_group_count == 1
        await scheduler.close()

    @pytest.mark.asyncio
    async def test_idempotent(self, scheduler: GroupScheduler) -> None:
        scheduler.ensure_timer("g1")
        scheduler.ensure_timer("g1")
        assert scheduler.active_group_count == 1
        await scheduler.close()

    @pytest.mark.asyncio
    async def test_multiple_groups(self, scheduler: GroupScheduler) -> None:
        scheduler.ensure_timer("g1")
        scheduler.ensure_timer("g2")
        assert scheduler.active_group_count == 2
        await scheduler.close()

    def test_no_timer_when_closed(self, scheduler: GroupScheduler) -> None:
        scheduler._closed = True
        scheduler.ensure_timer("g1")
        assert not scheduler.has_active_timer("g1")


class TestAggregateAndDecide:
    """聚合决策测试"""

    @pytest.mark.asyncio
    async def test_high_weight_direct_reply(
        self,
        scheduler: GroupScheduler,
        signal_queue: SignalQueue,
        reply_callback: AsyncMock,
    ) -> None:
        """高权重信号 → 直接触发回复"""
        s = _make_signal(weight=0.9)
        signal_queue.enqueue(s)
        signals = signal_queue.get_signals("g1")

        await scheduler._aggregate_and_decide("g1", signals)

        reply_callback.assert_called_once()
        decision: AggregatedDecision = reply_callback.call_args[0][0]
        assert decision.should_reply is True
        assert decision.group_id == "g1"
        assert decision.aggregated_weight >= 0.8
        assert decision.llm_confirmed is False

    @pytest.mark.asyncio
    async def test_low_weight_no_reply(
        self,
        scheduler: GroupScheduler,
        signal_queue: SignalQueue,
        reply_callback: AsyncMock,
    ) -> None:
        """低权重信号 → 不触发回复"""
        s = _make_signal(weight=0.2)
        signal_queue.enqueue(s)
        signals = signal_queue.get_signals("g1")

        await scheduler._aggregate_and_decide("g1", signals)

        reply_callback.assert_not_called()
        # 信号应被清除
        assert signal_queue.get_signals("g1") == []

    @pytest.mark.asyncio
    async def test_medium_weight_hybrid_mode_llm_confirm(
        self,
        config: ProactiveConfig,
        signal_queue: SignalQueue,
        reply_callback: AsyncMock,
        llm_callback: AsyncMock,
    ) -> None:
        """中等权重 + hybrid 模式 → LLM 确认"""
        config.proactive_mode = "hybrid"
        scheduler = GroupScheduler(
            config=config,
            signal_queue=signal_queue,
            on_reply=reply_callback,
            on_llm_confirm=llm_callback,
        )

        s = _make_signal(weight=0.6)
        signal_queue.enqueue(s)
        signals = signal_queue.get_signals("g1")

        await scheduler._aggregate_and_decide("g1", signals)

        llm_callback.assert_called_once()
        reply_callback.assert_called_once()
        decision: AggregatedDecision = reply_callback.call_args[0][0]
        assert decision.llm_confirmed is True

    @pytest.mark.asyncio
    async def test_medium_weight_hybrid_llm_declines(
        self,
        config: ProactiveConfig,
        signal_queue: SignalQueue,
        reply_callback: AsyncMock,
    ) -> None:
        """中等权重 + hybrid 模式 + LLM 拒绝 → 不回复"""
        config.proactive_mode = "hybrid"
        llm_decline = AsyncMock(return_value=False)
        scheduler = GroupScheduler(
            config=config,
            signal_queue=signal_queue,
            on_reply=reply_callback,
            on_llm_confirm=llm_decline,
        )

        s = _make_signal(weight=0.6)
        signal_queue.enqueue(s)
        signals = signal_queue.get_signals("g1")

        await scheduler._aggregate_and_decide("g1", signals)

        llm_decline.assert_called_once()
        reply_callback.assert_not_called()
        # 信号应被清除
        assert signal_queue.get_signals("g1") == []

    @pytest.mark.asyncio
    async def test_medium_weight_rule_mode_no_llm(
        self,
        scheduler: GroupScheduler,
        signal_queue: SignalQueue,
        reply_callback: AsyncMock,
    ) -> None:
        """中等权重 + rule 模式 → 不调用 LLM，权重不足直接跳过"""
        s = _make_signal(weight=0.6)
        signal_queue.enqueue(s)
        signals = signal_queue.get_signals("g1")

        await scheduler._aggregate_and_decide("g1", signals)

        # rule 模式下中等权重不够直接回复
        reply_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_reply_clears_signals(
        self,
        scheduler: GroupScheduler,
        signal_queue: SignalQueue,
        reply_callback: AsyncMock,
    ) -> None:
        """回复后清除信号"""
        s = _make_signal(weight=0.9)
        signal_queue.enqueue(s)
        signals = signal_queue.get_signals("g1")

        await scheduler._aggregate_and_decide("g1", signals)

        assert signal_queue.get_signals("g1") == []


class TestTimerLifecycle:
    """定时器生命周期测试"""

    @pytest.mark.asyncio
    async def test_timer_loop_silence_timeout(
        self,
        config: ProactiveConfig,
        signal_queue: SignalQueue,
    ) -> None:
        """沉默超时 → 定时器自动销毁"""
        config.signal_queue.check_interval_seconds = 1
        config.signal_queue.silence_timeout_seconds = 0  # 立即超时

        scheduler = GroupScheduler(
            config=config,
            signal_queue=signal_queue,
        )
        scheduler.ensure_timer("g1")
        # 不更新 last_message_time → silence = inf → 超时

        await asyncio.sleep(1.5)

        # 定时器应已销毁
        assert not scheduler.has_active_timer("g1")

        await scheduler.close()

    @pytest.mark.asyncio
    async def test_close_cancels_all_timers(self, scheduler: GroupScheduler) -> None:
        scheduler.ensure_timer("g1")
        scheduler.ensure_timer("g2")

        await scheduler.close()

        assert scheduler.active_group_count == 0

    @pytest.mark.asyncio
    async def test_no_reply_callback_warns(
        self,
        config: ProactiveConfig,
        signal_queue: SignalQueue,
    ) -> None:
        """无回复回调时 _execute_reply 不崩溃"""
        scheduler = GroupScheduler(
            config=config,
            signal_queue=signal_queue,
            on_reply=None,
        )
        s = _make_signal(weight=0.9)
        signal_queue.enqueue(s)
        signals = signal_queue.get_signals("g1")

        # 不应抛出异常
        await scheduler._aggregate_and_decide("g1", signals)

        await scheduler.close()


class TestLLMConfirmFallback:
    """LLM 确认降级测试"""

    @pytest.mark.asyncio
    async def test_llm_error_returns_false(
        self,
        config: ProactiveConfig,
        signal_queue: SignalQueue,
    ) -> None:
        """LLM 回调异常时返回 False"""
        error_callback = AsyncMock(side_effect=RuntimeError("LLM down"))
        scheduler = GroupScheduler(
            config=config,
            signal_queue=signal_queue,
            on_llm_confirm=error_callback,
        )

        result = await scheduler._try_llm_confirm("g1", [], [])
        assert result is False

        await scheduler.close()

    @pytest.mark.asyncio
    async def test_no_llm_callback_returns_false(
        self,
        config: ProactiveConfig,
        signal_queue: SignalQueue,
    ) -> None:
        """没有 LLM 回调时返回 False"""
        scheduler = GroupScheduler(
            config=config,
            signal_queue=signal_queue,
            on_llm_confirm=None,
        )

        result = await scheduler._try_llm_confirm("g1", [], [])
        assert result is False

        await scheduler.close()
