"""
语义置信度计算器测试
"""

import pytest

from iris_memory.capture.semantic.semantic_confidence import (
    SemanticConfidenceCalculator,
    ConfidenceResult,
)


class TestSemanticConfidenceCalculator:
    """置信度计算测试"""

    def setup_method(self):
        self.calc = SemanticConfidenceCalculator()

    def test_zero_evidence(self):
        """零证据返回最低置信度"""
        result = self.calc.calculate(0)
        assert result.confidence == self.calc.MIN_CONFIDENCE
        assert result.needs_human_review is True

    def test_single_evidence(self):
        """单个证据"""
        result = self.calc.calculate(1)
        assert result.confidence > self.calc.MIN_CONFIDENCE
        assert result.evidence_factor == pytest.approx(0.2)
        assert result.consistency_factor == 1.0

    def test_five_consistent_evidence(self):
        """5 条一致证据达到最大 evidence_factor"""
        result = self.calc.calculate(5, contradiction_count=0)
        assert result.evidence_factor == 1.0
        assert result.consistency_factor == 1.0
        assert result.confidence == self.calc.MAX_CONFIDENCE

    def test_more_than_five_evidence(self):
        """超过 5 条证据 evidence_factor 不超过 1"""
        result = self.calc.calculate(10, contradiction_count=0)
        assert result.evidence_factor == 1.0

    def test_with_contradictions(self):
        """有矛盾的记忆降低置信度"""
        # 3 条一致 + 1 条矛盾
        result = self.calc.calculate(3, contradiction_count=1)
        assert result.consistency_factor < 1.0
        assert result.consistency_factor == pytest.approx(0.75)
        # 置信度应该低于无矛盾的情况
        no_conflict = self.calc.calculate(3, contradiction_count=0)
        assert result.confidence < no_conflict.confidence

    def test_high_contradiction_ratio(self):
        """高矛盾比例导致低置信度"""
        result = self.calc.calculate(2, contradiction_count=3)
        assert result.confidence < self.calc.REVIEW_THRESHOLD
        assert result.needs_human_review is True

    def test_confidence_bounds(self):
        """置信度在合理范围内"""
        for evidence in range(0, 20):
            for contradiction in range(0, 5):
                result = self.calc.calculate(evidence, contradiction)
                assert self.calc.MIN_CONFIDENCE <= result.confidence <= self.calc.MAX_CONFIDENCE

    def test_review_threshold(self):
        """低置信度标记需要人工确认"""
        result = self.calc.calculate(1, contradiction_count=2)
        assert result.needs_human_review is True

    def test_calculate_from_memories(self):
        """从记忆列表计算"""
        from iris_memory.models.memory import Memory
        memories = [Memory(content="a"), Memory(content="b"), Memory(content="c")]
        result = self.calc.calculate_from_memories(
            memories,
            contradiction_ids=["x"],
        )
        assert result.evidence_count == 3
        assert result.contradiction_count == 1
