"""检索策略基类与参数封装"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from iris_memory.core.types import StorageLayer
    from iris_memory.models.emotion_state import EmotionalState
    from iris_memory.models.memory import Memory


@dataclass
class StrategyParams:
    """检索策略调用参数封装

    将 7 个独立参数合并为单一数据对象，减少方法签名冗余。
    """

    query: str
    user_id: str
    group_id: str | None = None
    top_k: int = 10
    emotional_state: EmotionalState | None = None
    storage_layer: StorageLayer | None = None
    persona_id: str | None = None


class RetrievalStrategyBase(Protocol):
    """检索策略协议

    所有策略实现此协议，由 MemoryRetrievalEngine 调度。
    """

    async def execute(self, params: StrategyParams) -> list[Memory]:
        """执行检索策略

        Args:
            params: 检索参数

        Returns:
            记忆列表
        """
        ...
