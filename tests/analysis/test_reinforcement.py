"""记忆强化引擎测试"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

from iris_memory.analysis.reinforcement import MemoryReinforcementEngine
from iris_memory.models.memory import Memory
from iris_memory.core.types import MemoryType, StorageLayer


@pytest.fixture
def mock_chroma():
    m = AsyncMock()
    m.get_memories_by_storage_layer = AsyncMock(return_value=[])
    m.get_memory = AsyncMock(return_value=None)
    m.update_memory = AsyncMock(return_value=True)
    m.get_active_user_ids = AsyncMock(return_value=[])
    return m


@pytest.fixture
def engine(mock_chroma):
    return MemoryReinforcementEngine(chroma_manager=mock_chroma)


@pytest.fixture
def sample_memories():
    """创建一批候选记忆"""
    now = datetime.now()
    return [
        Memory(
            id="mem_low_rif",
            type=MemoryType.FACT,
            content="用户喜欢Python",
            user_id="user_1",
            rif_score=0.2,
            importance_score=0.6,
            storage_layer=StorageLayer.EPISODIC,
            last_access_time=now - timedelta(days=20),
        ),
        Memory(
            id="mem_mid_rif",
            type=MemoryType.EMOTION,
            content="用户对学习很开心",
            user_id="user_1",
            rif_score=0.4,
            importance_score=0.7,
            storage_layer=StorageLayer.EPISODIC,
            last_access_time=now - timedelta(days=10),
        ),
        Memory(
            id="mem_high_rif",
            type=MemoryType.FACT,
            content="用户住在北京",
            user_id="user_1",
            rif_score=0.8,
            importance_score=0.5,
            storage_layer=StorageLayer.SEMANTIC,
            last_access_time=now - timedelta(days=2),
        ),
    ]


class TestSM2Reinforcement:
    """SM-2 变体强化逻辑测试"""

    @pytest.mark.asyncio
    async def test_reinforce_memory_updates_rif(self, engine, mock_chroma, sample_memories):
        """强化记忆应更新 RIF 评分"""
        mock_chroma.get_active_user_ids = AsyncMock(return_value=["user_1"])
        mock_chroma.get_memories_by_storage_layer = AsyncMock(
            side_effect=lambda layer: [m for m in sample_memories if m.storage_layer == layer]
        )

        old_rif = sample_memories[0].rif_score
        await engine._apply_sm2_reinforcement(sample_memories[0])

        assert sample_memories[0].rif_score > old_rif
        mock_chroma.update_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_reinforcement_cycle_updates_all_users(self, engine, mock_chroma, sample_memories):
        """强化循环应处理所有活跃用户"""
        mock_chroma.get_active_user_ids = AsyncMock(return_value=["user_1", "user_2"])
        mock_chroma.get_memories_by_storage_layer = AsyncMock(return_value=sample_memories)

        await engine._run_reinforcement_cycle()

        mock_chroma.get_active_user_ids.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_chroma_skips_cycle(self):
        """无 chroma 时跳过强化循环"""
        engine = MemoryReinforcementEngine(chroma_manager=None)
        await engine._run_reinforcement_cycle()


class TestCalculateNextReview:
    """下次回顾时间计算测试"""

    def test_first_review_interval(self, engine):
        """首次回顾间隔应为 1 天"""
        mem = Memory(
            id="m1", type=MemoryType.FACT, content="test",
            user_id="u1", storage_layer=StorageLayer.EPISODIC,
            access_count=1, importance_score=0.5,
        )
        last_review = datetime.now()
        next_review = engine._calculate_next_review(mem, last_review)
        assert (next_review - last_review).days == 1

    def test_second_review_interval(self, engine):
        """第二次回顾间隔应为 6 天"""
        mem = Memory(
            id="m2", type=MemoryType.FACT, content="test",
            user_id="u1", storage_layer=StorageLayer.EPISODIC,
            access_count=2, importance_score=0.5,
        )
        last_review = datetime.now()
        next_review = engine._calculate_next_review(mem, last_review)
        assert (next_review - last_review).days == 6

    def test_high_importance_shorter_interval(self, engine):
        """高重要性记忆应有更短的回顾间隔"""
        mem_low = Memory(
            id="m_low", type=MemoryType.FACT, content="test",
            user_id="u1", storage_layer=StorageLayer.EPISODIC,
            access_count=3, importance_score=0.3,
        )
        mem_high = Memory(
            id="m_high", type=MemoryType.FACT, content="test",
            user_id="u1", storage_layer=StorageLayer.EPISODIC,
            access_count=3, importance_score=0.9,
        )
        last_review = datetime.now()
        interval_low = (engine._calculate_next_review(mem_low, last_review) - last_review).days
        interval_high = (engine._calculate_next_review(mem_high, last_review) - last_review).days
        assert interval_high < interval_low

    def test_max_interval_capped(self, engine):
        """最大间隔应被限制为 90 天"""
        mem = Memory(
            id="m_max", type=MemoryType.FACT, content="test",
            user_id="u1", storage_layer=StorageLayer.EPISODIC,
            access_count=100, importance_score=0.1,
        )
        last_review = datetime.now()
        next_review = engine._calculate_next_review(mem, last_review)
        assert (next_review - last_review).days <= 90


class TestRecordReview:
    def test_record_increments_count(self, engine):
        engine.record_review("m1", "u1")
        engine.record_review("m2", "u1")
        assert engine._get_today_review_count("u1") == 2

    def test_separate_user_counts(self, engine):
        engine.record_review("m1", "u1")
        engine.record_review("m2", "u2")
        assert engine._get_today_review_count("u1") == 1
        assert engine._get_today_review_count("u2") == 1

    def test_get_last_review_time(self, engine):
        engine.record_review("m1", "u1")
        last_time = engine._get_last_review_time("m1", "u1")
        assert last_time is not None
        assert isinstance(last_time, datetime)

    def test_get_last_review_time_not_found(self, engine):
        last_time = engine._get_last_review_time("nonexistent", "u1")
        assert last_time is None


class TestLifecycle:
    """start() / stop() 生命周期管理测试"""

    @pytest.mark.asyncio
    async def test_start_creates_task(self, mock_chroma):
        engine = MemoryReinforcementEngine(chroma_manager=mock_chroma)
        await engine.start()
        assert engine._is_running is True
        assert engine._task is not None
        assert not engine._task.done()
        await engine.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, mock_chroma):
        engine = MemoryReinforcementEngine(chroma_manager=mock_chroma)
        await engine.start()
        await engine.stop()
        assert engine._is_running is False
        assert engine._task is None

    @pytest.mark.asyncio
    async def test_start_idempotent(self, mock_chroma):
        engine = MemoryReinforcementEngine(chroma_manager=mock_chroma)
        await engine.start()
        first_task = engine._task
        await engine.start()
        assert engine._task is first_task
        await engine.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(self, mock_chroma):
        engine = MemoryReinforcementEngine(chroma_manager=mock_chroma)
        await engine.stop()
        assert engine._is_running is False


class TestConfigIntegration:
    """配置联动测试"""

    def test_constructor_overrides_config(self):
        engine = MemoryReinforcementEngine(
            review_interval_hours=12,
        )
        assert engine._review_interval_hours == 12

    def test_default_config_fallback(self):
        """Without ConfigStore, defaults are used"""
        engine = MemoryReinforcementEngine()
        assert engine._review_interval_hours == MemoryReinforcementEngine.DEFAULT_REVIEW_INTERVAL_HOURS
