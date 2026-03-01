"""
Memory 模型语义提取字段测试
"""

import pytest
from datetime import datetime

from iris_memory.models.memory import Memory
from iris_memory.core.types import StorageLayer


class TestMemorySemanticFields:
    """Memory 模型新增语义提取字段测试"""

    def test_default_values(self):
        """默认值正确"""
        m = Memory()
        assert m.summarized is False
        assert m.semantic_memory_id is None
        assert m.evidence_ids == []
        assert m.source_type is None
        assert m.evidence_count == 0
        assert m.last_validated is None

    def test_set_semantic_fields(self):
        """设置语义提取字段"""
        m = Memory(
            summarized=True,
            semantic_memory_id="sem_123",
            evidence_ids=["m1", "m2", "m3"],
            source_type="semantic_extraction",
            evidence_count=3,
            last_validated=datetime(2025, 1, 1),
        )
        assert m.summarized is True
        assert m.semantic_memory_id == "sem_123"
        assert len(m.evidence_ids) == 3
        assert m.source_type == "semantic_extraction"
        assert m.evidence_count == 3
        assert m.last_validated == datetime(2025, 1, 1)

    def test_to_dict_includes_semantic_fields(self):
        """to_dict 序列化包含语义字段"""
        m = Memory(
            summarized=True,
            semantic_memory_id="sem_456",
            evidence_ids=["a", "b"],
            source_type="direct_upgrade",
            evidence_count=2,
            last_validated=datetime(2025, 6, 15, 12, 0, 0),
        )
        d = m.to_dict()
        assert d["summarized"] is True
        assert d["semantic_memory_id"] == "sem_456"
        assert d["evidence_ids"] == ["a", "b"]
        assert d["source_type"] == "direct_upgrade"
        assert d["evidence_count"] == 2
        assert "2025-06-15" in d["last_validated"]

    def test_from_dict_restores_semantic_fields(self):
        """from_dict 正确恢复语义字段"""
        m = Memory(
            summarized=True,
            semantic_memory_id="sem_789",
            evidence_ids=["x", "y"],
            source_type="semantic_extraction",
            evidence_count=2,
            last_validated=datetime(2025, 3, 1, 10, 30, 0),
        )
        d = m.to_dict()
        restored = Memory.from_dict(d)
        assert restored.summarized is True
        assert restored.semantic_memory_id == "sem_789"
        assert restored.evidence_ids == ["x", "y"]
        assert restored.source_type == "semantic_extraction"
        assert restored.evidence_count == 2
        assert restored.last_validated == datetime(2025, 3, 1, 10, 30, 0)

    def test_from_dict_backward_compatible(self):
        """旧数据（无语义字段）兼容"""
        d = {
            "id": "old_mem",
            "content": "hello",
            "user_id": "u1",
            "type": "fact",
            "storage_layer": "episodic",
        }
        m = Memory.from_dict(d)
        assert m.summarized is False
        assert m.semantic_memory_id is None
        assert m.evidence_ids == []
        assert m.source_type is None
        assert m.evidence_count == 0
        assert m.last_validated is None
