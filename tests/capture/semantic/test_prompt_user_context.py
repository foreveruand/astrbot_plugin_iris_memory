"""
LLM prompt 用户上下文测试

验证 SEMANTIC_EXTRACTION_PROMPT 包含 {user_id} 占位符，
并且 _extract_cluster 正确传递 user_id。
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from iris_memory.models.memory import Memory
from iris_memory.core.types import MemoryType, StorageLayer
from iris_memory.capture.semantic.semantic_extractor import (
    SemanticExtractor,
    SEMANTIC_EXTRACTION_PROMPT,
)
from iris_memory.capture.semantic.semantic_clustering import MemoryCluster
from iris_memory.utils.llm_helper import LLMCallResult


class TestPromptUserContext:
    """测试 LLM 提取 prompt 包含用户上下文"""

    def test_prompt_has_user_id_placeholder(self):
        """Prompt 模板包含 {user_id} 占位符"""
        assert "{user_id}" in SEMANTIC_EXTRACTION_PROMPT

    def test_prompt_format_with_user_id(self):
        """Prompt 格式化后包含实际的 user_id"""
        formatted = SEMANTIC_EXTRACTION_PROMPT.format(
            user_id="test_user_123",
            cluster_key="火锅",
            memories_text="- 记忆1: ...",
        )
        assert "test_user_123" in formatted
        assert "用户ID" in formatted

    @pytest.mark.asyncio
    async def test_extract_passes_user_id_from_cluster(self):
        """_extract_cluster 使用 cluster.user_id"""
        extractor = SemanticExtractor()
        extractor._llm_provider = Mock()
        extractor._llm_resolved_id = "test"

        captured_prompts = []

        async def mock_call_llm(_ctx, _provider, _pid, prompt, **kwargs):
            captured_prompts.append(prompt)
            return LLMCallResult(
                success=True,
                content='{"content":"test","type":"fact","subtype":"","contradiction_ids":[]}',
                parsed_json={
                    "content": "test",
                    "type": "fact",
                    "subtype": "",
                    "contradiction_ids": [],
                },
            )

        memories = [
            Memory(
                id=f"m{i}",
                content=f"content{i}",
                user_id="user_xyz",
                storage_layer=StorageLayer.EPISODIC,
                created_time=datetime.now() - timedelta(days=60),
            )
            for i in range(3)
        ]
        cluster = MemoryCluster(
            cluster_id="kw_1",
            cluster_key="test_key",
            cluster_type="entity",
            memories=memories,
            user_id="user_xyz",
        )

        with patch("iris_memory.capture.semantic.semantic_extractor.call_llm", side_effect=mock_call_llm):
            await extractor._extract_cluster(cluster)

        assert len(captured_prompts) == 1
        assert "user_xyz" in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_extract_falls_back_to_ref_memory_user_id(self):
        """当 cluster.user_id 为空时，使用 ref_memory.user_id"""
        extractor = SemanticExtractor()
        extractor._llm_provider = Mock()
        extractor._llm_resolved_id = "test"

        captured_prompts = []

        async def mock_call_llm(_ctx, _provider, _pid, prompt, **kwargs):
            captured_prompts.append(prompt)
            return LLMCallResult(
                success=True,
                content='{"content":"test","type":"fact","subtype":"","contradiction_ids":[]}',
                parsed_json={
                    "content": "test",
                    "type": "fact",
                    "subtype": "",
                    "contradiction_ids": [],
                },
            )

        memories = [
            Memory(
                id="m1",
                content="content",
                user_id="fallback_user",
                storage_layer=StorageLayer.EPISODIC,
                created_time=datetime.now() - timedelta(days=60),
            )
        ]
        cluster = MemoryCluster(
            cluster_id="kw_1",
            cluster_key="test",
            cluster_type="entity",
            memories=memories,
            user_id="",  # empty user_id
        )

        with patch("iris_memory.capture.semantic.semantic_extractor.call_llm", side_effect=mock_call_llm):
            await extractor._extract_cluster(cluster)

        assert len(captured_prompts) == 1
        assert "fallback_user" in captured_prompts[0]
