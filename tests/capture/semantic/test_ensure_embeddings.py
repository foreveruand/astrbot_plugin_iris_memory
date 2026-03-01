"""
嵌入向量降级测试

验证 _ensure_embeddings：
- 为缺少 embedding 的记忆生成嵌入
- 已有 embedding 的记忆不重复生成
- embedding_manager 不可用时平滑跳过
- 单条嵌入失败不影响其他记忆
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from iris_memory.models.memory import Memory
from iris_memory.core.types import StorageLayer
from iris_memory.capture.semantic.semantic_extractor import SemanticExtractor


def _make_memory(
    memory_id: str = "m1",
    content: str = "test content",
    embedding: np.ndarray | None = None,
) -> Memory:
    m = Memory(
        id=memory_id,
        content=content,
        storage_layer=StorageLayer.EPISODIC,
        created_time=datetime.now() - timedelta(days=60),
        user_id="user_1",
    )
    m.embedding = embedding
    return m


class TestEnsureEmbeddings:
    """_ensure_embeddings 方法测试"""

    @pytest.mark.asyncio
    async def test_generates_missing_embeddings(self):
        """为缺少 embedding 的记忆生成嵌入向量"""
        mock_embed = AsyncMock(return_value=[1.0, 0.0, 0.0])
        mock_embedding_mgr = MagicMock()
        mock_embedding_mgr.embed = mock_embed

        mock_chroma = MagicMock()
        mock_chroma.embedding_manager = mock_embedding_mgr

        extractor = SemanticExtractor(chroma_manager=mock_chroma)
        memories = [
            _make_memory("m1", "content 1"),  # no embedding
            _make_memory("m2", "content 2"),  # no embedding
        ]

        await extractor._ensure_embeddings(memories)

        assert mock_embed.call_count == 2
        assert memories[0].embedding is not None
        assert memories[1].embedding is not None
        np.testing.assert_array_equal(memories[0].embedding, np.array([1.0, 0.0, 0.0]))

    @pytest.mark.asyncio
    async def test_skips_existing_embeddings(self):
        """已有 embedding 的记忆不重新生成"""
        mock_embed = AsyncMock(return_value=[1.0, 0.0, 0.0])
        mock_embedding_mgr = MagicMock()
        mock_embedding_mgr.embed = mock_embed

        mock_chroma = MagicMock()
        mock_chroma.embedding_manager = mock_embedding_mgr

        extractor = SemanticExtractor(chroma_manager=mock_chroma)
        existing = np.array([0.0, 1.0, 0.0])
        memories = [
            _make_memory("m1", "content 1", embedding=existing),
            _make_memory("m2", "content 2"),  # no embedding
        ]

        await extractor._ensure_embeddings(memories)

        # Only m2 should have been embedded
        assert mock_embed.call_count == 1
        np.testing.assert_array_equal(memories[0].embedding, existing)  # unchanged

    @pytest.mark.asyncio
    async def test_no_chroma_manager_noop(self):
        """chroma_manager 不可用时不报错"""
        extractor = SemanticExtractor(chroma_manager=None)
        memories = [_make_memory("m1", "content")]

        # Should not raise
        await extractor._ensure_embeddings(memories)
        assert memories[0].embedding is None

    @pytest.mark.asyncio
    async def test_no_embedding_manager_noop(self):
        """embedding_manager 不可用时不报错"""
        mock_chroma = MagicMock(spec=[])  # no embedding_manager attr
        extractor = SemanticExtractor(chroma_manager=mock_chroma)
        memories = [_make_memory("m1", "content")]

        await extractor._ensure_embeddings(memories)
        assert memories[0].embedding is None

    @pytest.mark.asyncio
    async def test_partial_failure_continues(self):
        """单条嵌入失败不影响其他记忆"""
        call_count = 0

        async def flaky_embed(text):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("embed failed")
            return [1.0, 0.0, 0.0]

        mock_embedding_mgr = MagicMock()
        mock_embedding_mgr.embed = flaky_embed

        mock_chroma = MagicMock()
        mock_chroma.embedding_manager = mock_embedding_mgr

        extractor = SemanticExtractor(chroma_manager=mock_chroma)
        memories = [
            _make_memory("m1", "fail"),
            _make_memory("m2", "succeed"),
        ]

        await extractor._ensure_embeddings(memories)

        # m1 failed, should remain None
        assert memories[0].embedding is None
        # m2 succeeded
        assert memories[1].embedding is not None

    @pytest.mark.asyncio
    async def test_empty_content_skipped(self):
        """空 content 的记忆不尝试生成嵌入"""
        mock_embed = AsyncMock(return_value=[1.0, 0.0, 0.0])
        mock_embedding_mgr = MagicMock()
        mock_embedding_mgr.embed = mock_embed

        mock_chroma = MagicMock()
        mock_chroma.embedding_manager = mock_embedding_mgr

        extractor = SemanticExtractor(chroma_manager=mock_chroma)
        memories = [
            _make_memory("m1", ""),  # empty content
            _make_memory("m2", "has content"),
        ]

        await extractor._ensure_embeddings(memories)

        # Only m2 should be embedded
        assert mock_embed.call_count == 1

    @pytest.mark.asyncio
    async def test_all_have_embeddings_noop(self):
        """所有记忆都有 embedding 时不调用 embed"""
        mock_embed = AsyncMock(return_value=[1.0, 0.0, 0.0])
        mock_embedding_mgr = MagicMock()
        mock_embedding_mgr.embed = mock_embed

        mock_chroma = MagicMock()
        mock_chroma.embedding_manager = mock_embedding_mgr

        extractor = SemanticExtractor(chroma_manager=mock_chroma)
        memories = [
            _make_memory("m1", "content", embedding=np.array([1.0, 0.0])),
            _make_memory("m2", "content", embedding=np.array([0.0, 1.0])),
        ]

        await extractor._ensure_embeddings(memories)

        assert mock_embed.call_count == 0
