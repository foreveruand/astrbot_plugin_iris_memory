"""
语义提取器测试
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from iris_memory.models.memory import Memory
from iris_memory.core.types import MemoryType, StorageLayer, QualityLevel
from iris_memory.capture.semantic.semantic_extractor import (
    SemanticExtractor,
    ExtractionResult,
    SEMANTIC_EXTRACTION_PROMPT,
    BATCH_SEMANTIC_EXTRACTION_PROMPT,
)
from iris_memory.capture.semantic.semantic_clustering import MemoryCluster
from iris_memory.capture.semantic.semantic_confidence import SemanticConfidenceCalculator
from iris_memory.utils.llm_helper import LLMCallResult


def _make_memory(
    content: str = "test",
    memory_id: str = "mem_1",
    days_old: int = 60,
    memory_type: MemoryType = MemoryType.FACT,
    confidence: float = 0.6,
) -> Memory:
    m = Memory(
        id=memory_id,
        content=content,
        type=memory_type,
        confidence=confidence,
        storage_layer=StorageLayer.EPISODIC,
        created_time=datetime.now() - timedelta(days=days_old),
        user_id="user_1",
        group_id="group_1",
        persona_id="default",
    )
    return m


def _make_cluster(memories: list[Memory] | None = None) -> MemoryCluster:
    if memories is None:
        memories = [
            _make_memory("今天吃了火锅", "m1", 60),
            _make_memory("和朋友去吃了重庆火锅", "m2", 50),
            _make_memory("想吃火锅了", "m3", 40),
        ]
    return MemoryCluster(
        cluster_id="kw_1",
        cluster_key="火锅",
        cluster_type="entity",
        memories=memories,
    )


class TestSemanticExtractorInit:
    """初始化测试"""

    def test_default_init(self):
        extractor = SemanticExtractor()
        assert extractor.chroma_manager is None
        assert extractor.clustering is not None
        assert extractor.confidence_calculator is not None

    def test_set_chroma_manager(self):
        extractor = SemanticExtractor()
        mock_cm = Mock()
        extractor.set_chroma_manager(mock_cm)
        assert extractor.chroma_manager is mock_cm


class TestFormatMemories:
    """Prompt 格式化测试"""

    def test_format_memories_for_prompt(self):
        memories = [_make_memory("火锅好吃", "m1")]
        text = SemanticExtractor._format_memories_for_prompt(memories)
        assert "m1" in text
        assert "火锅好吃" in text
        assert "记忆1" in text


class TestExtractCluster:
    """单个聚类提取测试"""

    @pytest.mark.asyncio
    async def test_extract_cluster_success(self):
        """成功提取"""
        extractor = SemanticExtractor()
        extractor._llm_provider = Mock()
        extractor._llm_resolved_id = "test_provider"

        llm_result = LLMCallResult(
            success=True,
            content='{"content":"喜欢吃火锅","type":"fact","subtype":"preference","contradiction_ids":[]}',
            parsed_json={
                "content": "喜欢吃火锅",
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
        assert result.semantic_memory.content == "喜欢吃火锅"
        assert result.semantic_memory.storage_layer == StorageLayer.SEMANTIC
        assert result.semantic_memory.source_type == "semantic_extraction"
        assert result.semantic_memory.evidence_count == 3
        assert len(result.semantic_memory.evidence_ids) == 3

    @pytest.mark.asyncio
    async def test_extract_cluster_llm_failure(self):
        """LLM 调用失败"""
        extractor = SemanticExtractor()
        extractor._llm_provider = Mock()
        extractor._llm_resolved_id = "test_provider"

        llm_result = LLMCallResult(success=False, error="timeout")

        with patch("iris_memory.capture.semantic.semantic_extractor.call_llm", return_value=llm_result):
            cluster = _make_cluster()
            result = await extractor._extract_cluster(cluster)

        assert result.success is False
        assert result.semantic_memory is None

    @pytest.mark.asyncio
    async def test_extract_cluster_empty_content(self):
        """LLM 返回空 content"""
        extractor = SemanticExtractor()
        extractor._llm_provider = Mock()
        extractor._llm_resolved_id = "test_provider"

        llm_result = LLMCallResult(
            success=True,
            content='{"content":"","type":"fact","subtype":"","contradiction_ids":[]}',
            parsed_json={"content": "", "type": "fact", "subtype": "", "contradiction_ids": []},
        )

        with patch("iris_memory.capture.semantic.semantic_extractor.call_llm", return_value=llm_result):
            cluster = _make_cluster()
            result = await extractor._extract_cluster(cluster)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_extract_cluster_with_contradictions(self):
        """有矛盾记忆的提取"""
        extractor = SemanticExtractor()
        extractor._llm_provider = Mock()
        extractor._llm_resolved_id = "test_provider"

        llm_result = LLMCallResult(
            success=True,
            content='{"content":"对火锅态度复杂","type":"emotion","subtype":"preference","contradiction_ids":["m3"]}',
            parsed_json={
                "content": "对火锅态度复杂",
                "type": "emotion",
                "subtype": "preference",
                "contradiction_ids": ["m3"],
            },
        )

        with patch("iris_memory.capture.semantic.semantic_extractor.call_llm", return_value=llm_result):
            cluster = _make_cluster()
            result = await extractor._extract_cluster(cluster)

        assert result.success is True
        assert result.confidence_result.contradiction_count == 1
        assert result.confidence_result.confidence < SemanticConfidenceCalculator.MAX_CONFIDENCE


class TestPersistResult:
    """持久化测试"""

    @pytest.mark.asyncio
    async def test_persist_success(self):
        """成功持久化"""
        mock_chroma = AsyncMock()
        mock_chroma.add_memory = AsyncMock(return_value="new_id")
        mock_chroma.update_memory = AsyncMock(return_value=True)

        extractor = SemanticExtractor(chroma_manager=mock_chroma)

        cluster = _make_cluster()
        semantic_memory = Memory(
            content="喜欢吃火锅",
            storage_layer=StorageLayer.SEMANTIC,
        )

        from iris_memory.capture.semantic.semantic_confidence import ConfidenceResult
        conf = ConfidenceResult(
            confidence=0.85,
            evidence_count=3,
            contradiction_count=0,
            evidence_factor=0.6,
            consistency_factor=1.0,
            needs_human_review=False,
        )

        result = ExtractionResult(
            cluster=cluster,
            semantic_memory=semantic_memory,
            confidence_result=conf,
            success=True,
        )

        persisted = await extractor._persist_result(result)
        assert persisted is True
        mock_chroma.add_memory.assert_called_once_with(semantic_memory)
        # 3 source memories should be updated
        assert mock_chroma.update_memory.call_count == 3

    @pytest.mark.asyncio
    async def test_persist_marks_source_memories(self):
        """持久化后源记忆被标记"""
        mock_chroma = AsyncMock()
        mock_chroma.add_memory = AsyncMock(return_value="new_id")
        mock_chroma.update_memory = AsyncMock(return_value=True)

        extractor = SemanticExtractor(chroma_manager=mock_chroma, source_expiry_days=90)

        cluster = _make_cluster()
        semantic_memory = Memory(
            id="semantic_1",
            content="喜欢吃火锅",
            storage_layer=StorageLayer.SEMANTIC,
        )

        from iris_memory.capture.semantic.semantic_confidence import ConfidenceResult
        conf = ConfidenceResult(
            confidence=0.85, evidence_count=3, contradiction_count=0,
            evidence_factor=0.6, consistency_factor=1.0, needs_human_review=False,
        )

        result = ExtractionResult(
            cluster=cluster,
            semantic_memory=semantic_memory,
            confidence_result=conf,
            success=True,
        )

        await extractor._persist_result(result)

        # 检查源记忆被标记
        for mem in cluster.memories:
            assert mem.summarized is True
            assert mem.semantic_memory_id == "semantic_1"
            assert mem.expires_at is not None


class TestRunFullPipeline:
    """完整流程测试"""

    @pytest.mark.asyncio
    async def test_run_no_chroma(self):
        """无 ChromaManager 时跳过"""
        extractor = SemanticExtractor()
        results = await extractor.run()
        assert results == []

    @pytest.mark.asyncio
    async def test_run_no_episodic_memories(self):
        """无 EPISODIC 记忆时跳过"""
        mock_chroma = AsyncMock()
        mock_chroma.get_memories_by_storage_layer = AsyncMock(return_value=[])

        extractor = SemanticExtractor(chroma_manager=mock_chroma)
        results = await extractor.run()
        assert results == []

    @pytest.mark.asyncio
    async def test_run_full_pipeline(self):
        """完整流程：获取记忆 → 聚类 → 提取 → 持久化"""
        memories = [
            _make_memory(f"火锅{i}", f"m{i}", days_old=40 + i, confidence=0.6)
            for i in range(3)
        ]
        for m in memories:
            m.keywords = ["火锅"]

        mock_chroma = AsyncMock()
        mock_chroma.get_memories_by_storage_layer = AsyncMock(return_value=memories)
        mock_chroma.add_memory = AsyncMock(return_value="new_id")
        mock_chroma.update_memory = AsyncMock(return_value=True)

        extractor = SemanticExtractor(chroma_manager=mock_chroma)
        extractor._llm_provider = Mock()
        extractor._llm_resolved_id = "test"

        llm_result = LLMCallResult(
            success=True,
            content='{"content":"喜欢吃火锅","type":"fact","subtype":"preference","contradiction_ids":[]}',
            parsed_json={
                "content": "喜欢吃火锅",
                "type": "fact",
                "subtype": "preference",
                "contradiction_ids": [],
            },
        )

        with patch("iris_memory.capture.semantic.semantic_extractor.call_llm", return_value=llm_result):
            results = await extractor.run()

        # 应该有至少1个结果
        assert len(results) >= 1
        success_results = [r for r in results if r.success]
        assert len(success_results) >= 1


class TestBatchExtraction:
    """批量聚类提取测试"""

    @pytest.mark.asyncio
    async def test_batch_extraction_multiple_clusters(self):
        """多个聚类合并为1次LLM调用"""
        extractor = SemanticExtractor(batch_size=5)
        extractor._llm_provider = Mock()
        extractor._llm_resolved_id = "test_provider"

        clusters = [
            _make_cluster([
                _make_memory("今天吃了火锅", "m1", 60),
                _make_memory("火锅太辣了", "m2", 50),
            ]),
            MemoryCluster(
                cluster_id="kw_2",
                cluster_key="咖啡",
                cluster_type="entity",
                memories=[
                    _make_memory("喝了杯拿铁", "m3", 40),
                    _make_memory("咖啡好苦", "m4", 30),
                ],
            ),
        ]

        batch_response = LLMCallResult(
            success=True,
            content="[]",
            parsed_json=[
                {"content": "喜欢吃火锅", "type": "fact", "subtype": "preference", "contradiction_ids": []},
                {"content": "关注咖啡口味", "type": "fact", "subtype": "interest", "contradiction_ids": []},
            ],
        )

        with patch("iris_memory.capture.semantic.semantic_extractor.call_llm", return_value=batch_response) as mock_llm:
            results = await extractor.extract_from_clusters(clusters)

        # 只调用1次LLM（批量）
        assert mock_llm.call_count == 1
        assert len(results) == 2
        assert results[0].success is True
        assert results[0].semantic_memory.content == "喜欢吃火锅"
        assert results[1].success is True
        assert results[1].semantic_memory.content == "关注咖啡口味"

    @pytest.mark.asyncio
    async def test_batch_extraction_single_cluster_no_batch(self):
        """单个聚类不走批量路径"""
        extractor = SemanticExtractor(batch_size=5)
        extractor._llm_provider = Mock()
        extractor._llm_resolved_id = "test_provider"

        clusters = [_make_cluster()]

        single_response = LLMCallResult(
            success=True,
            content="{}",
            parsed_json={"content": "喜欢吃火锅", "type": "fact", "subtype": "preference", "contradiction_ids": []},
        )

        with patch("iris_memory.capture.semantic.semantic_extractor.call_llm", return_value=single_response) as mock_llm:
            results = await extractor.extract_from_clusters(clusters)

        assert mock_llm.call_count == 1
        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_batch_extraction_fallback_on_failure(self):
        """批量调用失败时降级为逐个提取"""
        extractor = SemanticExtractor(batch_size=5)
        extractor._llm_provider = Mock()
        extractor._llm_resolved_id = "test_provider"

        clusters = [
            _make_cluster([
                _make_memory("今天吃了火锅", "m1", 60),
                _make_memory("火锅太辣了", "m2", 50),
            ]),
            MemoryCluster(
                cluster_id="kw_2",
                cluster_key="咖啡",
                cluster_type="entity",
                memories=[
                    _make_memory("喝了杯拿铁", "m3", 40),
                    _make_memory("咖啡好苦", "m4", 30),
                ],
            ),
        ]

        # 第一次调用（批量）失败，后续逐个调用成功
        fail_result = LLMCallResult(success=False, error="timeout")
        ok_result = LLMCallResult(
            success=True,
            content="{}",
            parsed_json={"content": "语义内容", "type": "fact", "subtype": "preference", "contradiction_ids": []},
        )

        call_count = {"n": 0}
        async def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return fail_result
            return ok_result

        with patch("iris_memory.capture.semantic.semantic_extractor.call_llm", side_effect=side_effect) as mock_llm:
            results = await extractor.extract_from_clusters(clusters)

        # 1次批量失败 + 2次逐个 = 3次
        assert mock_llm.call_count == 3
        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_batch_size_splits_correctly(self):
        """超过batch_size的聚类被分成多个批次"""
        extractor = SemanticExtractor(batch_size=2)
        extractor._llm_provider = Mock()
        extractor._llm_resolved_id = "test_provider"

        clusters = [
            MemoryCluster(
                cluster_id=f"kw_{i}",
                cluster_key=f"topic_{i}",
                cluster_type="entity",
                memories=[_make_memory(f"内容{i}", f"m{i}", 60)],
            )
            for i in range(5)
        ]

        batch_response = LLMCallResult(
            success=True,
            content="[]",
            parsed_json=[
                {"content": f"语义{i}", "type": "fact", "subtype": "preference", "contradiction_ids": []}
                for i in range(2)
            ],
        )
        single_response = LLMCallResult(
            success=True,
            content="{}",
            parsed_json={"content": "语义x", "type": "fact", "subtype": "preference", "contradiction_ids": []},
        )

        call_count = {"n": 0}
        async def side_effect(*args, **kwargs):
            call_count["n"] += 1
            # 批次1(2个), 批次2(2个), 批次3(1个=不走batch)
            if call_count["n"] <= 2:
                return batch_response
            return single_response

        with patch("iris_memory.capture.semantic.semantic_extractor.call_llm", side_effect=side_effect):
            results = await extractor.extract_from_clusters(clusters)

        # batch_size=2 → 5个聚类: [2, 2, 1] → 2次批量 + 1次单个 = 3次调用
        assert call_count["n"] == 3
        assert len(results) == 5

    def test_parse_batch_response_array(self):
        """解析数组响应"""
        extractor = SemanticExtractor()
        result = LLMCallResult(
            success=True,
            parsed_json=[{"content": "a"}, {"content": "b"}],
        )
        items = extractor._parse_batch_response(result, 2)
        assert items is not None
        assert len(items) == 2

    def test_parse_batch_response_dict_with_results_key(self):
        """解析包含 results key 的字典响应"""
        extractor = SemanticExtractor()
        result = LLMCallResult(
            success=True,
            parsed_json={"results": [{"content": "a"}]},
        )
        items = extractor._parse_batch_response(result, 1)
        assert items is not None
        assert len(items) == 1

    def test_parse_batch_response_failure(self):
        """解析失败返回 None"""
        extractor = SemanticExtractor()
        result = LLMCallResult(success=False, error="fail")
        items = extractor._parse_batch_response(result, 2)
        assert items is None

    def test_build_single_result_empty_content(self):
        """空 content 构建失败结果"""
        extractor = SemanticExtractor()
        cluster = _make_cluster()
        result = extractor._build_single_result(cluster, {"content": "", "type": "fact"})
        assert result.success is False

    def test_batch_size_default(self):
        """默认 batch_size"""
        extractor = SemanticExtractor()
        assert extractor.batch_size == SemanticExtractor.DEFAULT_BATCH_SIZE

    def test_batch_size_minimum(self):
        """batch_size 最小为 1"""
        extractor = SemanticExtractor(batch_size=0)
        assert extractor.batch_size == 1
