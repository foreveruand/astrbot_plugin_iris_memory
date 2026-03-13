"""
宽限期管理器

在记忆被最终删除前进行智能评估，自动决定保留或清除。
"""

from datetime import datetime, timedelta
from typing import List, Optional

from iris_memory.core.types import QualityLevel, StorageLayer
from iris_memory.models.memory import Memory
from iris_memory.utils.logger import get_logger

logger = get_logger("grace_period")


class GracePeriodManager:
    """宽限期管理器

    评估即将清除的记忆，根据记忆特征自动决定保留、进入宽限期或直接清除。
    """

    DEFAULT_GRACE_DAYS = 7
    SILENT_DELETE_CONFIDENCE_THRESHOLD = 0.3
    SILENT_DELETE_ACCESS_THRESHOLD = 0
    
    AUTO_KEEP_EMOTIONAL_WEIGHT_THRESHOLD = 0.5
    AUTO_KEEP_IMPORTANCE_THRESHOLD = 0.6
    AUTO_KEEP_ACCESS_THRESHOLD = 2

    def __init__(
        self,
        chroma_manager=None,
        proactive_manager=None,
        grace_days: int = DEFAULT_GRACE_DAYS,
    ):
        self._chroma = chroma_manager
        self._proactive = proactive_manager
        self._grace_days = grace_days

    async def evaluate_and_apply(self, memory: Memory) -> str:
        """评估记忆是否应保留、进入宽限期或直接清除。

        Returns:
            "protected"       - 受保护，跳过
            "auto_keep"       - 自动保留（高价值记忆）
            "grace_period"    - 进入宽限期
            "silent_delete"   - 直接清除（极低价值）
            "already_pending" - 已在宽限期中
            "expired"         - 宽限期已到
        """
        if hasattr(memory, "is_protected") and memory.is_protected:
            return "protected"
        if memory.is_user_requested:
            return "protected"
        if memory.quality_level == QualityLevel.CONFIRMED:
            return "protected"

        if memory.grace_period_expires_at is not None:
            if datetime.now() >= memory.grace_period_expires_at:
                return "expired"
            return "already_pending"

        if (
            memory.confidence < self.SILENT_DELETE_CONFIDENCE_THRESHOLD
            and memory.access_count <= self.SILENT_DELETE_ACCESS_THRESHOLD
            and memory.emotional_weight < 0.3
        ):
            return "silent_delete"

        if self._should_auto_keep(memory):
            logger.debug(f"Memory {memory.id[:8]} auto-kept (emotional={memory.emotional_weight:.2f}, importance={memory.importance_score:.2f})")
            return "auto_keep"

        await self._initiate_grace_period(memory)
        return "grace_period"

    def _should_auto_keep(self, memory: Memory) -> bool:
        """判断记忆是否应自动保留（无需宽限期等待）
        
        高价值记忆特征：
        - 情感权重 >= 0.5（有情感价值）
        - 或 重要性 >= 0.6 且 访问 >= 2 次（有持续关注）
        - 或 置信度 >= 0.6 且 访问 >= 1 次（质量较高）
        """
        if memory.emotional_weight >= self.AUTO_KEEP_EMOTIONAL_WEIGHT_THRESHOLD:
            return True
        if memory.importance_score >= self.AUTO_KEEP_IMPORTANCE_THRESHOLD and memory.access_count >= self.AUTO_KEEP_ACCESS_THRESHOLD:
            return True
        if memory.confidence >= 0.6 and memory.access_count >= 1:
            return True
        return False

    async def _initiate_grace_period(self, memory: Memory) -> None:
        """设置宽限期"""
        memory.grace_period_expires_at = datetime.now() + timedelta(days=self._grace_days)
        memory.review_status = "pending_review"

        if self._chroma:
            try:
                await self._chroma.update_memory(memory)
            except Exception as e:
                logger.warning(f"Failed to persist grace period for {memory.id}: {e}")

    async def get_pending_review_memories(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Memory]:
        """获取当前处于宽限期的记忆列表
        
        Args:
            user_id: 用户ID（None 表示所有用户）
            group_id: 群组ID
            limit: 最大返回数量
        """
        if not self._chroma:
            return []
        try:
            # 从 EPISODIC 层查询所有候选
            episodic = await self._chroma.get_memories_by_storage_layer(StorageLayer.EPISODIC)
            if not episodic:
                return []
            pending = [
                m for m in episodic
                if m.review_status == "pending_review"
                and m.grace_period_expires_at is not None
                and (user_id is None or m.user_id == user_id)
                and (group_id is None or m.group_id == group_id)
            ]
            pending.sort(key=lambda m: m.grace_period_expires_at)
            return pending[:limit]
        except Exception as e:
            logger.warning(f"Failed to get pending review memories: {e}")
            return []
