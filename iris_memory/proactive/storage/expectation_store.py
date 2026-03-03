"""
FollowUpExpectation 存储

内存存储，按 group_id 索引，支持 CRUD 和过期清理。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from iris_memory.proactive.models import FollowUpExpectation
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.expectation_store")


class ExpectationStore:
    """FollowUp 期待存储

    内存中按 group_id 存储活跃的 FollowUpExpectation。
    每个群最多一个活跃期待（后创建的覆盖前一个）。
    """

    def __init__(self) -> None:
        # group_id -> FollowUpExpectation
        self._expectations: Dict[str, FollowUpExpectation] = {}

    def put(self, expectation: FollowUpExpectation) -> None:
        """存储或覆盖期待

        Args:
            expectation: 期待对象
        """
        old = self._expectations.get(expectation.group_id)
        if old:
            logger.debug(
                f"Replacing expectation for group {expectation.group_id}: "
                f"{old.expectation_id} -> {expectation.expectation_id}"
            )
        self._expectations[expectation.group_id] = expectation

    def get(self, group_id: str) -> Optional[FollowUpExpectation]:
        """获取某群的活跃期待

        自动检查过期，如果已过期则移除并返回 None。

        Args:
            group_id: 群组 ID

        Returns:
            FollowUpExpectation 或 None
        """
        exp = self._expectations.get(group_id)
        if exp is None:
            return None

        if exp.is_window_expired:
            logger.debug(
                f"Expectation {exp.expectation_id} for group {group_id} "
                f"window expired, removing"
            )
            del self._expectations[group_id]
            return None

        return exp

    def remove(self, group_id: str) -> Optional[FollowUpExpectation]:
        """移除某群的期待

        Args:
            group_id: 群组 ID

        Returns:
            被移除的期待，或 None
        """
        return self._expectations.pop(group_id, None)

    def has_active(self, group_id: str) -> bool:
        """检查某群是否有活跃（未过期）期待

        Args:
            group_id: 群组 ID
        """
        exp = self.get(group_id)
        return exp is not None

    def get_all(self) -> List[FollowUpExpectation]:
        """获取所有活跃期待"""
        # 先清理过期的
        expired_groups = [
            g for g, e in self._expectations.items() if e.is_window_expired
        ]
        for g in expired_groups:
            del self._expectations[g]

        return list(self._expectations.values())

    def clear(self) -> int:
        """清除所有期待

        Returns:
            清除的数量
        """
        count = len(self._expectations)
        self._expectations.clear()
        return count

    @property
    def count(self) -> int:
        """活跃期待数量"""
        return len(self._expectations)
