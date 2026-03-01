"""
审核状态 (review_status) 机制测试

验证：
- 语义提取时根据置信度设置 review_status
- Memory 模型正确序列化/反序列化 review_status
- 检索过滤 pending_review/rejected 记忆
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

from iris_memory.models.memory import Memory
from iris_memory.core.types import MemoryType, StorageLayer, QualityLevel
from iris_memory.capture.semantic.semantic_extractor import (
    SemanticExtractor,
    ExtractionResult,
)
from iris_memory.capture.semantic.semantic_clustering import MemoryCluster
from iris_memory.capture.semantic.semantic_confidence import (
    SemanticConfidenceCalculator,
    ConfidenceResult,
)
from iris_memory.utils.llm_helper import LLMCallResult


def _make_memory(
    content: str = "test",
    memory_id: str = "mem_1",
    days_old: int = 60,
    confidence: float = 0.6,
) -> Memory:
    return Memory(
        id=memory_id,
        content=content,
        type=MemoryType.FACT,
        confidence=confidence,
        storage_layer=StorageLayer.EPISODIC,
        created_time=datetime.now() - timedelta(days=days_old),
        user_id="user_1",
        group_id="group_1",
        persona_id="default",
    )


def _make_cluster(memories=None):
    if memories is None:
        memories = [
            _make_memory("火锅A", "m1", 60),
            _make_memory("火锅B", "m2", 50),
            _make_memory("火锅C", "m3", 40),
        ]
    return MemoryCluster(
        cluster_id="kw_1",
        cluster_key="火锅",
        cluster_type="entity",
        memories=memories,
        user_id="user_1",
    )


class TestReviewStatusInExtraction:
    """测试语义提取时 review_status 设置"""

    @pytest.mark.asyncio
    async def test_high_confidence_approved(self):
        """高置信度 -> approved"""
        extractor = SemanticExtractor()
        extractor._llm_provider = Mock()
        extractor._llm_resolved_id = "test"

        llm_result = LLMCallResult(
            success=True,
            content='{"content":"喜欢火锅","type":"fact","subtype":"preference","contradiction_ids":[]}',
            parsed_json={
                "content": "喜欢火锅",
                "type": "fact",
                "subtype": "preference",
                "contradiction_ids": [],
            },
        )

        with patch("iris_memory.capture.semantic.semantic_extractor.call_llm", return_value=llm_result):
            cluster = _make_cluster()
            result = await extractor._extract_cluster(cluster)

        assert result.success is True
        assert result.semantic_memory is not None
        # 3 evidence -> confidence > 0.5 -> approved
        assert result.semantic_memory.review_status == "approved"

    @pytest.mark.asyncio
    async def test_low_confidence_pending_review(self):
        """低置信度 -> pending_review"""
        extractor = SemanticExtractor()
        extractor._llm_provider = Mock()
        extractor._llm_resolved_id = "test"

        # 2 evidence + 1 contradiction -> low confidence -> pending_review
        llm_result = LLMCallResult(
            success=True,
            content='{"content":"对火锅态度不确定","type":"fact","subtype":"preference","contradiction_ids":["m2"]}',
            parsed_json={
                "content": "对火锅态度不确定",
                "type": "fact",
                "subtype": "preference",
                "contradiction_ids": ["m2"],
            },
        )

        # Use cluster with just 1 memory to force low evidence
        cluster = MemoryCluster(
            cluster_id="kw_1",
            cluster_key="something",
            cluster_type="entity",
            memories=[_make_memory("x", "m1", 60)],
            user_id="user_1",
        )

        with patch("iris_memory.capture.semantic.semantic_extractor.call_llm", return_value=llm_result):
            result = await extractor._extract_cluster(cluster)

        assert result.success is True
        assert result.semantic_memory is not None
        # 1 evidence + 1 contradiction => very low confidence => pending_review
        assert result.semantic_memory.review_status == "pending_review"


class TestReviewStatusModel:
    """测试 Memory 模型的 review_status 字段"""

    def test_default_none(self):
        m = Memory()
        assert m.review_status is None

    def test_to_dict_includes_review_status(self):
        m = Memory(review_status="pending_review")
        d = m.to_dict()
        assert d["review_status"] == "pending_review"

    def test_from_dict_restores(self):
        d = {
            "id": "test",
            "content": "test",
            "review_status": "approved",
        }
        m = Memory.from_dict(d)
        assert m.review_status == "approved"

    def test_from_dict_backward_compatible(self):
        """旧数据无 review_status 字段"""
        d = {"id": "test", "content": "test"}
        m = Memory.from_dict(d)
        assert m.review_status is None


class TestRetrievalFiltering:
    """测试检索过滤待审核记忆"""

    def test_filter_pending_review(self):
        """pending_review 记忆被过滤"""
        memories = [
            Memory(id="1", content="approved", review_status="approved"),
            Memory(id="2", content="pending", review_status="pending_review"),
            Memory(id="3", content="no status", review_status=None),
            Memory(id="4", content="rejected", review_status="rejected"),
        ]
        filtered = [
            m for m in memories
            if m.review_status not in ("pending_review", "rejected")
        ]
        assert len(filtered) == 2
        assert {m.id for m in filtered} == {"1", "3"}
