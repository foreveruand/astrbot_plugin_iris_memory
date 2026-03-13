"""
记忆强化引擎

基于间隔重复效应 (SM-2 变体)，定期分析重要记忆并更新 RIF 评分，
防止高价值陪伴记忆因长期不被访问而衰减。
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from iris_memory.core.types import StorageLayer
from iris_memory.models.memory import Memory
from iris_memory.utils.logger import get_logger

logger = get_logger("reinforcement")


class MemoryReinforcementEngine:
    """记忆强化引擎

    独立后台服务，定期分析重要记忆并更新 RIF 评分。
    通过 start()/stop() 管理生命周期。
    """

    DEFAULT_REVIEW_INTERVAL_HOURS = 6
    MIN_REVIEW_INTERVAL_HOURS = 4

    def __init__(
        self,
        chroma_manager: Any = None,
        review_interval_hours: Optional[int] = None,
    ):
        self._chroma = chroma_manager
        self._review_interval_hours = (
            review_interval_hours
            if review_interval_hours is not None
            else self._load_config_interval()
        )
        self._review_history: Dict[str, List[tuple]] = {}
        self._task: Optional[asyncio.Task] = None
        self._is_running = False

    async def start(self) -> None:
        """启动后台强化循环"""
        if self._is_running:
            return
        self._is_running = True
        self._task = asyncio.create_task(self._reinforcement_loop())
        logger.info(f"ReinforcementEngine started: interval={self._review_interval_hours}h")

    async def stop(self) -> None:
        """停止后台强化循环"""
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"ReinforcementEngine stop error: {e}")
            self._task = None
        logger.debug("ReinforcementEngine stopped")

    async def _reinforcement_loop(self) -> None:
        """强化调度循环，每 review_interval_hours 触发一次扫描"""
        interval_seconds = self._review_interval_hours * 3600
        while self._is_running:
            try:
                await asyncio.sleep(interval_seconds)
                if not self._is_running:
                    break
                await self._run_reinforcement_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Reinforcement loop error: {e}")

    async def _run_reinforcement_cycle(self) -> None:
        """执行一次强化扫描：分析重要记忆并更新 RIF 评分"""
        if not self._chroma:
            return

        try:
            active_user_ids = await self._chroma.get_active_user_ids()
        except Exception as e:
            logger.warning(f"Failed to get active users: {e}")
            return

        for user_id in active_user_ids:
            try:
                await self._update_rif_scores(user_id)
            except Exception as e:
                logger.warning(f"RIF update failed for user {user_id}: {e}")

    async def _update_rif_scores(self, user_id: str) -> None:
        """更新用户的记忆 RIF 评分"""
        try:
            episodic = await self._chroma.get_memories_by_storage_layer(StorageLayer.EPISODIC)
            semantic = await self._chroma.get_memories_by_storage_layer(StorageLayer.SEMANTIC)
            all_memories = (episodic or []) + (semantic or [])

            candidates = [
                m for m in all_memories
                if m.user_id == user_id and m.importance_score >= 0.4
            ]

            for memory in candidates:
                await self._apply_sm2_reinforcement(memory)

        except Exception as e:
            logger.warning(f"Failed to update RIF scores: {e}")

    async def _apply_sm2_reinforcement(self, memory: Memory) -> None:
        """应用 SM-2 变体强化：根据间隔重复效应更新 RIF 评分"""
        last_review = self._get_last_review_time(memory.id, memory.user_id)
        next_review = self._calculate_next_review(memory, last_review or memory.created_at)

        if datetime.now() >= next_review:
            old_rif = memory.rif_score
            self._reinforce_memory(memory)
            self.record_review(memory.id, memory.user_id)
            try:
                await self._chroma.update_memory(memory)
                logger.debug(f"RIF updated: {memory.id} {old_rif:.2f} -> {memory.rif_score:.2f}")
            except Exception as e:
                logger.warning(f"Failed to update memory {memory.id}: {e}")

    def _reinforce_memory(self, memory: Memory) -> None:
        """强化记忆：更新 RIF 评分"""
        memory.update_access()
        memory.rif_score = min(1.0, memory.rif_score + 0.1)

    def _calculate_next_review(self, memory: Memory, last_review: datetime) -> datetime:
        """SM-2 变体计算下次回顾时间"""
        review_count = memory.access_count
        ef = 2.5 - memory.importance_score * 0.8
        ef = max(1.3, ef)

        if review_count <= 1:
            interval_days = 1
        elif review_count == 2:
            interval_days = 6
        else:
            interval_days = int(6 * (ef ** (review_count - 2)))

        interval_days = min(interval_days, 90)
        return last_review + timedelta(days=interval_days)

    @staticmethod
    def _load_config_interval() -> int:
        """从 ConfigStore 读取回顾间隔小时数"""
        try:
            from iris_memory.config import get_store
            return get_store().get(
                "memory.reinforcement.interval_hours",
                MemoryReinforcementEngine.DEFAULT_REVIEW_INTERVAL_HOURS,
            )
        except Exception:
            return MemoryReinforcementEngine.DEFAULT_REVIEW_INTERVAL_HOURS

    def record_review(self, memory_id: str, user_id: str) -> None:
        """记录一次回顾"""
        if user_id not in self._review_history:
            self._review_history[user_id] = []
        self._review_history[user_id].append((memory_id, datetime.now()))

    def _get_last_review_time(self, memory_id: str, user_id: str) -> Optional[datetime]:
        history = self._review_history.get(user_id, [])
        for mid, t in reversed(history):
            if mid == memory_id:
                return t
        return None
