"""
连续回复限制测试

测试内容：
1. _is_consecutive_limit_reached 基本逻辑
2. _record_reply_time 记录行为
3. 窗口过期自动清理
4. handle_batch 集成：达到限制后跳过
5. 跨会话隔离
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

from iris_memory.proactive.proactive_manager import ProactiveReplyManager
from iris_memory.proactive.proactive_reply_detector import (
    ProactiveReplyDecision,
    ReplyUrgency,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def decision_should_reply() -> ProactiveReplyDecision:
    return ProactiveReplyDecision(
        should_reply=True,
        urgency=ReplyUrgency.HIGH,
        reason="test_reason",
        suggested_delay=0,
        reply_context={"emotion": {"primary": "joy", "intensity": 0.6}},
    )


@pytest.fixture
def mock_reply_detector(decision_should_reply):
    detector = Mock()
    detector.analyze = AsyncMock(return_value=decision_should_reply)
    return detector


@pytest.fixture
def mock_context_with_queue():
    context = Mock()
    context._event_queue = asyncio.Queue()
    return context


@pytest_asyncio.fixture
async def manager(mock_reply_detector, mock_context_with_queue):
    mgr = ProactiveReplyManager(
        astrbot_context=mock_context_with_queue,
        reply_detector=mock_reply_detector,
        config={
            "enable_proactive_reply": True,
            "reply_cooldown": 60,
            "max_daily_replies": 100,
        },
    )
    await mgr.initialize()
    yield mgr
    await mgr.stop()


# =============================================================================
# 基本逻辑测试
# =============================================================================

class TestConsecutiveLimitBasic:
    """连续回复限制基本逻辑"""

    def test_not_reached_when_empty(self, manager):
        """无记录时不应达到限制"""
        assert manager._is_consecutive_limit_reached("u1:private") is False

    def test_not_reached_below_max(self, manager):
        """未达到最大次数时不应限制"""
        now = time.time()
        manager._recent_replies["u1:private"] = [now - 10, now - 5]
        assert manager._is_consecutive_limit_reached("u1:private") is False

    def test_reached_at_max(self, manager):
        """达到最大次数时应限制"""
        now = time.time()
        manager._recent_replies["u1:private"] = [now - 30, now - 20, now - 10]
        assert manager._is_consecutive_limit_reached("u1:private") is True

    def test_reached_above_max(self, manager):
        """超过最大次数时应限制"""
        now = time.time()
        manager._recent_replies["u1:private"] = [
            now - 40, now - 30, now - 20, now - 10
        ]
        assert manager._is_consecutive_limit_reached("u1:private") is True

    def test_expired_entries_cleaned(self, manager):
        """过期记录应被自动清理"""
        now = time.time()
        # 3条记录，但全部过期（超过 CONSECUTIVE_WINDOW=300s）
        manager._recent_replies["u1:private"] = [
            now - 400, now - 350, now - 310
        ]
        assert manager._is_consecutive_limit_reached("u1:private") is False
        # 验证清理后只保留窗口内的记录
        assert len(manager._recent_replies["u1:private"]) == 0

    def test_mixed_expired_and_active(self, manager):
        """混合过期和有效记录时只计算有效记录"""
        now = time.time()
        manager._recent_replies["u1:private"] = [
            now - 400,  # 过期
            now - 350,  # 过期
            now - 30,   # 有效
            now - 10,   # 有效
        ]
        # 只有2条有效记录，未达到限制3
        assert manager._is_consecutive_limit_reached("u1:private") is False
        assert len(manager._recent_replies["u1:private"]) == 2


class TestRecordReplyTime:
    """记录回复时间测试"""

    def test_record_creates_entry(self, manager):
        """首次记录应创建条目"""
        manager._record_reply_time("u1:private")
        assert len(manager._recent_replies["u1:private"]) == 1

    def test_record_appends(self, manager):
        """多次记录应追加"""
        manager._record_reply_time("u1:private")
        manager._record_reply_time("u1:private")
        assert len(manager._recent_replies["u1:private"]) == 2

    def test_record_cleans_expired(self, manager):
        """记录时应清理过期条目"""
        now = time.time()
        manager._recent_replies["u1:private"] = [now - 400]  # 过期
        manager._record_reply_time("u1:private")
        # 过期条目已清理，只有新记录
        assert len(manager._recent_replies["u1:private"]) == 1
        assert manager._recent_replies["u1:private"][0] >= now


class TestSessionIsolation:
    """会话隔离测试"""

    def test_different_sessions_independent(self, manager):
        """不同会话的计数互相独立"""
        now = time.time()
        manager._recent_replies["u1:private"] = [now - 30, now - 20, now - 10]
        manager._recent_replies["u2:private"] = [now - 5]

        assert manager._is_consecutive_limit_reached("u1:private") is True
        assert manager._is_consecutive_limit_reached("u2:private") is False

    def test_private_and_group_independent(self, manager):
        """私聊和群聊计数互相独立"""
        now = time.time()
        manager._recent_replies["u1:private"] = [now - 30, now - 20, now - 10]
        manager._recent_replies["u1:g1"] = [now - 5]

        assert manager._is_consecutive_limit_reached("u1:private") is True
        assert manager._is_consecutive_limit_reached("u1:g1") is False


class TestCustomLimits:
    """自定义限制参数测试"""

    def test_custom_max_consecutive(self, manager):
        """自定义最大连续次数"""
        manager.MAX_CONSECUTIVE_REPLIES = 5
        now = time.time()
        manager._recent_replies["u1:private"] = [
            now - 40, now - 30, now - 20, now - 10
        ]
        # 4次 < 5，不应限制
        assert manager._is_consecutive_limit_reached("u1:private") is False

    def test_custom_window(self, manager):
        """自定义时间窗口"""
        manager.CONSECUTIVE_WINDOW = 60  # 缩短为1分钟
        now = time.time()
        manager._recent_replies["u1:private"] = [
            now - 90,   # 过期
            now - 70,   # 过期
            now - 30,   # 有效
        ]
        # 只有1条有效记录
        assert manager._is_consecutive_limit_reached("u1:private") is False


# =============================================================================
# handle_batch 集成测试
# =============================================================================

class TestHandleBatchConsecutiveLimit:
    """handle_batch 中的连续回复限制集成测试"""

    @pytest.mark.asyncio
    async def test_handle_batch_skips_when_limit_reached(self, manager):
        """达到连续限制时 handle_batch 应跳过"""
        now = time.time()
        session_key = "u1:private"
        manager._recent_replies[session_key] = [now - 30, now - 20, now - 10]

        await manager.handle_batch(messages=["你好"], user_id="u1")
        assert manager.pending_tasks.qsize() == 0
        assert manager.stats["replies_consecutive_limited"] == 1

    @pytest.mark.asyncio
    async def test_handle_batch_allows_when_under_limit(self, manager):
        """未达到限制时 handle_batch 应正常处理"""
        now = time.time()
        session_key = "u1:private"
        manager._recent_replies[session_key] = [now - 30]

        await manager.handle_batch(messages=["你好"], user_id="u1")
        assert manager.pending_tasks.qsize() == 1

    @pytest.mark.asyncio
    async def test_process_task_records_reply_time(
        self, mock_reply_detector, mock_context_with_queue, decision_should_reply
    ):
        """任务处理成功后应记录回复时间"""
        mgr = ProactiveReplyManager(
            astrbot_context=mock_context_with_queue,
            reply_detector=mock_reply_detector,
            config={"enable_proactive_reply": True},
        )
        await mgr.initialize()

        try:
            with patch("iris_memory.proactive.proactive_manager.ProactiveMessageEvent") as event_cls:
                event_cls.return_value = Mock()
                await mgr.handle_batch(
                    messages=["test"], user_id="u1", umo="test:FriendMessage:u1"
                )
                # 等待任务处理完成
                await asyncio.sleep(0.2)
                # 应该有记录
                assert len(mgr._recent_replies.get("u1:private", [])) == 1
        finally:
            await mgr.stop()

    @pytest.mark.asyncio
    async def test_stat_counter_increments(self, manager):
        """连续限制计数器应正确递增"""
        now = time.time()
        manager._recent_replies["u1:private"] = [now - 30, now - 20, now - 10]

        await manager.handle_batch(messages=["你好"], user_id="u1")
        await manager.handle_batch(messages=["在吗"], user_id="u1")
        assert manager.stats["replies_consecutive_limited"] == 2
