"""
语义提取置信度计算

根据方案中的公式：
semantic_confidence = base × evidence_factor × consistency_factor

其中：
- base = 0.5
- evidence_factor = min(1.0, evidence_count / 5)
- consistency_factor = 1.0 - contradiction_ratio

最终结果映射到 [0.3, 0.95] 区间。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from iris_memory.utils.logger import get_logger

logger = get_logger("semantic_confidence")


@dataclass(frozen=True, slots=True)
class ConfidenceResult:
    """置信度计算结果"""
    confidence: float
    evidence_count: int
    contradiction_count: int
    evidence_factor: float
    consistency_factor: float
    needs_human_review: bool  # 置信度 < 0.5 时建议人工确认


class SemanticConfidenceCalculator:
    """语义记忆置信度计算器"""

    BASE_CONFIDENCE: float = 0.5
    EVIDENCE_FULL_COUNT: int = 5  # 达到此数量 evidence_factor 为 1.0
    MIN_CONFIDENCE: float = 0.3
    MAX_CONFIDENCE: float = 0.95
    REVIEW_THRESHOLD: float = 0.5

    def calculate(
        self,
        evidence_count: int,
        contradiction_count: int = 0,
    ) -> ConfidenceResult:
        """计算语义记忆的置信度

        Args:
            evidence_count: 支持证据数量（参与聚类的记忆数）
            contradiction_count: 矛盾记忆数量

        Returns:
            ConfidenceResult
        """
        if evidence_count <= 0:
            return ConfidenceResult(
                confidence=self.MIN_CONFIDENCE,
                evidence_count=0,
                contradiction_count=contradiction_count,
                evidence_factor=0.0,
                consistency_factor=1.0,
                needs_human_review=True,
            )

        evidence_factor = min(1.0, evidence_count / self.EVIDENCE_FULL_COUNT)

        total = evidence_count + contradiction_count
        contradiction_ratio = contradiction_count / total if total > 0 else 0.0
        consistency_factor = 1.0 - contradiction_ratio

        raw = self.BASE_CONFIDENCE * evidence_factor * consistency_factor

        # 映射到 [MIN, MAX] 区间
        # raw 的理论范围是 [0, 0.5]，映射到 [0.3, 0.95]
        confidence = self.MIN_CONFIDENCE + (raw / self.BASE_CONFIDENCE) * (
            self.MAX_CONFIDENCE - self.MIN_CONFIDENCE
        )
        confidence = max(self.MIN_CONFIDENCE, min(self.MAX_CONFIDENCE, confidence))

        needs_review = confidence < self.REVIEW_THRESHOLD

        return ConfidenceResult(
            confidence=round(confidence, 4),
            evidence_count=evidence_count,
            contradiction_count=contradiction_count,
            evidence_factor=round(evidence_factor, 4),
            consistency_factor=round(consistency_factor, 4),
            needs_human_review=needs_review,
        )

    def calculate_from_memories(
        self,
        memories: List,
        contradiction_ids: List[str] | None = None,
    ) -> ConfidenceResult:
        """从记忆列表直接计算

        Args:
            memories: 参与聚类的记忆列表
            contradiction_ids: 被判定为矛盾的记忆 ID 集

        Returns:
            ConfidenceResult
        """
        evidence_count = len(memories)
        contradiction_count = len(contradiction_ids) if contradiction_ids else 0
        return self.calculate(evidence_count, contradiction_count)
