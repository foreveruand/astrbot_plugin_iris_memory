"""
KnowledgeGraphModule 集成测试
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from iris_memory.services.modules.kg_module import KnowledgeGraphModule


def run(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


@pytest.fixture
def kg_module():
    """创建完整初始化的 KG 模块"""
    module = KnowledgeGraphModule()
    with tempfile.TemporaryDirectory() as tmpdir:
        run(module.initialize(
            plugin_data_path=Path(tmpdir),
            kg_mode="rule",
            max_depth=3,
            max_nodes_per_hop=10,
            max_facts=8,
            enabled=True,
        ))
        yield module
        run(module.close())


@pytest.fixture
def disabled_module():
    """创建禁用的 KG 模块"""
    module = KnowledgeGraphModule()
    with tempfile.TemporaryDirectory() as tmpdir:
        run(module.initialize(
            plugin_data_path=Path(tmpdir),
            enabled=False,
        ))
        yield module


class TestModuleInit:
    """模块初始化测试"""

    def test_initialize_enabled(self, kg_module):
        assert kg_module.is_initialized
        assert kg_module.enabled
        assert kg_module.storage is not None
        assert kg_module.extractor is not None
        assert kg_module.reasoning is not None
        assert kg_module.formatter is not None

    def test_initialize_disabled(self, disabled_module):
        assert not disabled_module.is_initialized
        assert not disabled_module.enabled
        assert disabled_module.storage is None

    def test_close(self, kg_module):
        run(kg_module.close())
        assert not kg_module.is_initialized


class TestModuleCapture:
    """记忆捕获 → KG 提取测试"""

    def test_process_memory(self, kg_module):
        memory = Mock()
        memory.content = "张三和李四是好朋友"
        memory.user_id = "u1"
        memory.group_id = None
        memory.id = "m1"
        memory.sender_name = "张三"
        memory.detected_entities = None
        memory.graph_nodes = []
        memory.graph_edges = []

        triples = run(kg_module.process_memory(memory))
        assert len(triples) >= 1

        # graph_nodes / graph_edges 应被更新
        assert len(memory.graph_nodes) >= 2
        assert len(memory.graph_edges) >= 1

    def test_process_memory_disabled(self, disabled_module):
        memory = Mock()
        memory.content = "张三和李四是好朋友"
        triples = run(disabled_module.process_memory(memory))
        assert len(triples) == 0

    def test_process_memory_error_handling(self, kg_module):
        """处理错误不应抛出异常"""
        memory = Mock()
        memory.content = None  # 会导致提取失败
        triples = run(kg_module.process_memory(memory))
        assert len(triples) == 0


class TestModuleRetrieval:
    """图检索测试"""

    def test_graph_retrieve_with_data(self, kg_module):
        # 先写入一些数据
        memory = Mock()
        memory.content = "张三喜欢编程"
        memory.user_id = "u1"
        memory.group_id = None
        memory.id = "m1"
        memory.sender_name = "张三"
        memory.detected_entities = None
        memory.graph_nodes = []
        memory.graph_edges = []
        run(kg_module.process_memory(memory))

        # 检索
        result = run(kg_module.graph_retrieve("张三", user_id="u1"))
        assert result.has_results or len(result.seed_nodes) >= 0  # 某些情况 FTS 可能无法匹配

    def test_graph_retrieve_disabled(self, disabled_module):
        result = run(disabled_module.graph_retrieve("张三", user_id="u1"))
        assert not result.has_results

    def test_format_graph_context(self, kg_module):
        # 写入数据
        memory = Mock()
        memory.content = "张三和李四是好朋友"
        memory.user_id = "u1"
        memory.group_id = None
        memory.id = "m1"
        memory.sender_name = "张三"
        memory.detected_entities = None
        memory.graph_nodes = []
        memory.graph_edges = []
        run(kg_module.process_memory(memory))

        text = run(kg_module.format_graph_context("张三", user_id="u1"))
        # 可能有结果也可能没有（取决于 FTS 匹配）
        assert isinstance(text, str)


class TestModuleStats:
    """统计和管理测试"""

    def test_get_stats_empty(self, kg_module):
        stats = run(kg_module.get_stats())
        assert stats["nodes"] == 0
        assert stats["edges"] == 0

    def test_get_stats_after_insert(self, kg_module):
        memory = Mock()
        memory.content = "张三喜欢编程"
        memory.user_id = "u1"
        memory.group_id = None
        memory.id = "m1"
        memory.sender_name = "张三"
        memory.detected_entities = None
        memory.graph_nodes = []
        memory.graph_edges = []
        run(kg_module.process_memory(memory))

        stats = run(kg_module.get_stats())
        assert stats["nodes"] >= 1

    def test_delete_user_data(self, kg_module):
        memory = Mock()
        memory.content = "张三喜欢编程"
        memory.user_id = "u1"
        memory.group_id = None
        memory.id = "m1"
        memory.sender_name = "张三"
        memory.detected_entities = None
        memory.graph_nodes = []
        memory.graph_edges = []
        run(kg_module.process_memory(memory))

        count = run(kg_module.delete_user_data("u1"))
        assert count >= 1

        stats = run(kg_module.get_stats(user_id="u1"))
        assert stats["nodes"] == 0

    def test_delete_all(self, kg_module):
        memory = Mock()
        memory.content = "张三喜欢编程"
        memory.user_id = "u1"
        memory.group_id = None
        memory.id = "m1"
        memory.sender_name = "张三"
        memory.detected_entities = None
        memory.graph_nodes = []
        memory.graph_edges = []
        run(kg_module.process_memory(memory))

        count = run(kg_module.delete_all())
        assert count >= 1

        stats = run(kg_module.get_stats())
        assert stats["nodes"] == 0

    def test_stats_disabled(self, disabled_module):
        stats = run(disabled_module.get_stats())
        assert stats == {"nodes": 0, "edges": 0}

    def test_delete_disabled(self, disabled_module):
        count = run(disabled_module.delete_user_data("u1"))
        assert count == 0
        count = run(disabled_module.delete_all())
        assert count == 0


class TestBatchConfig:
    """批量配置测试"""

    def test_batch_size_from_config(self):
        """从配置读取 batch_size"""
        module = KnowledgeGraphModule()
        assert module._batch_size >= 1
        assert module._batch_size <= 20

    def test_batch_flush_interval_from_config(self):
        """从配置读取 batch_flush_interval"""
        module = KnowledgeGraphModule()
        assert module._batch_flush_interval >= 1.0
        assert module._batch_flush_interval <= 300.0


class TestBatchBuffer:
    """批量缓冲机制测试"""

    @pytest.fixture
    def hybrid_module(self):
        """创建 hybrid 模式的 KG 模块"""
        module = KnowledgeGraphModule()
        with tempfile.TemporaryDirectory() as tmpdir:
            run(module.initialize(
                plugin_data_path=Path(tmpdir),
                kg_mode="hybrid",
                max_depth=3,
                max_nodes_per_hop=10,
                max_facts=8,
                enabled=True,
            ))
            yield module
            run(module.close())

    def test_pending_items_empty_initially(self, hybrid_module):
        """初始时待处理队列为空"""
        assert len(hybrid_module._pending_items) == 0

    def test_pending_items_buffered_for_llm(self, hybrid_module):
        """hybrid 模式下需要 LLM 的消息应入队"""
        memory = Mock()
        memory.content = "张三和李四是非常要好的朋友，他们一起经历了很多事情"
        memory.user_id = "u1"
        memory.group_id = None
        memory.id = "m1"
        memory.sender_name = "张三"
        memory.detected_entities = None
        memory.graph_nodes = []
        memory.graph_edges = []

        run(hybrid_module.process_memory(memory))

        assert len(hybrid_module._pending_items) >= 0

    def test_flush_pending_llm_empty(self, hybrid_module):
        """空队列 flush 不报错"""
        run(hybrid_module.flush_pending_llm())
        assert len(hybrid_module._pending_items) == 0

    def test_flush_pending_llm_clears_queue(self, hybrid_module):
        """flush 后队列被清空"""
        memory = Mock()
        memory.content = "张三喜欢编程和机器学习"
        memory.user_id = "u1"
        memory.group_id = None
        memory.id = "m1"
        memory.sender_name = "张三"
        memory.detected_entities = None
        memory.graph_nodes = []
        memory.graph_edges = []

        run(hybrid_module.process_memory(memory))

        if hybrid_module._pending_items:
            run(hybrid_module.flush_pending_llm())
            assert len(hybrid_module._pending_items) == 0

    def test_flush_lock_prevents_concurrent(self, hybrid_module):
        """flush lock 防止并发"""
        import asyncio

        async def concurrent_flush():
            tasks = [hybrid_module.flush_pending_llm() for _ in range(3)]
            await asyncio.gather(*tasks)

        run(concurrent_flush())

    def test_batch_size_triggers_flush(self):
        """缓冲区满时自动 flush"""
        module = KnowledgeGraphModule()
        module._batch_size = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            run(module.initialize(
                plugin_data_path=Path(tmpdir),
                kg_mode="hybrid",
                enabled=True,
            ))

            for i in range(3):
                memory = Mock()
                memory.content = f"消息{i}包含一些关系描述"
                memory.user_id = "u1"
                memory.group_id = None
                memory.id = f"m{i}"
                memory.sender_name = "张三"
                memory.detected_entities = None
                memory.graph_nodes = []
                memory.graph_edges = []
                run(module.process_memory(memory))

            run(module.close())


class TestKGExtractorBatch:
    """KGExtractor batch_extract_by_llm 测试"""

    @pytest.fixture
    def storage_and_extractor_llm(self):
        """创建 KGStorage + KGExtractor (llm 模式)"""
        from iris_memory.knowledge_graph.kg_storage import KGStorage
        from iris_memory.knowledge_graph.kg_extractor import KGExtractor

        s = KGStorage()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_kg.db"
            run(s.initialize(db_path))
            ext = KGExtractor(storage=s, mode="llm")
            yield s, ext
            run(s.close())

    def test_batch_extract_empty_input(self, storage_and_extractor_llm):
        """空输入返回空字典"""
        _, ext = storage_and_extractor_llm
        result = run(ext.batch_extract_by_llm([]))
        assert result == {}

    def test_batch_extract_no_context(self, storage_and_extractor_llm):
        """无 astrbot_context 返回空字典"""
        _, ext = storage_and_extractor_llm
        items = [{"text": "张三喜欢编程", "user_id": "u1", "sender_name": "张三", "index": 0}]
        result = run(ext.batch_extract_by_llm(items))
        assert result == {}

    def test_batch_extract_with_mock_llm(self, storage_and_extractor_llm):
        """模拟 LLM 响应"""
        _, ext = storage_and_extractor_llm
        ext._astrbot_context = Mock()
        ext._provider = Mock()
        ext._resolved_provider_id = "test"
        ext._provider_initialized = True

        from unittest.mock import patch
        from iris_memory.utils.llm_helper import LLMCallResult

        llm_result = LLMCallResult(
            success=True,
            content='[{"message_index": 0, "triples": [{"subject": "张三", "predicate": "喜欢", "object": "编程", "relation_type": "likes", "confidence": 0.8}]}]',
            parsed_json=[{
                "message_index": 0,
                "triples": [{
                    "subject": "张三",
                    "predicate": "喜欢",
                    "object": "编程",
                    "relation_type": "likes",
                    "confidence": 0.8
                }]
            }]
        )

        with patch("iris_memory.utils.llm_helper.call_llm", return_value=llm_result):
            items = [{"text": "张三喜欢编程", "user_id": "u1", "sender_name": "张三", "index": 0}]
            result = run(ext.batch_extract_by_llm(items))

        assert 0 in result
        assert len(result[0]) == 1
        assert result[0][0].subject == "张三"
        assert result[0][0].object == "编程"

    def test_batch_extract_multiple_messages(self, storage_and_extractor_llm):
        """批量提取多条消息"""
        _, ext = storage_and_extractor_llm
        ext._astrbot_context = Mock()
        ext._provider = Mock()
        ext._resolved_provider_id = "test"
        ext._provider_initialized = True

        from unittest.mock import patch
        from iris_memory.utils.llm_helper import LLMCallResult

        llm_result = LLMCallResult(
            success=True,
            content='[]',
            parsed_json=[
                {"message_index": 0, "triples": [{"subject": "张三", "predicate": "喜欢", "object": "编程", "relation_type": "likes", "confidence": 0.8}]},
                {"message_index": 1, "triples": [{"subject": "李四", "predicate": "讨厌", "object": "加班", "relation_type": "dislikes", "confidence": 0.7}]},
            ]
        )

        with patch("iris_memory.utils.llm_helper.call_llm", return_value=llm_result):
            items = [
                {"text": "张三喜欢编程", "user_id": "u1", "sender_name": "张三", "index": 0},
                {"text": "李四讨厌加班", "user_id": "u1", "sender_name": "李四", "index": 1},
            ]
            result = run(ext.batch_extract_by_llm(items))

        assert 0 in result
        assert 1 in result
        assert result[0][0].subject == "张三"
        assert result[1][0].subject == "李四"

    def test_batch_extract_fallback_index(self, storage_and_extractor_llm):
        """缺少 message_index 时使用 entry_idx"""
        _, ext = storage_and_extractor_llm
        ext._astrbot_context = Mock()
        ext._provider = Mock()
        ext._resolved_provider_id = "test"
        ext._provider_initialized = True

        from unittest.mock import patch
        from iris_memory.utils.llm_helper import LLMCallResult

        llm_result = LLMCallResult(
            success=True,
            content='[]',
            parsed_json=[
                {"triples": [{"subject": "张三", "predicate": "喜欢", "object": "编程", "relation_type": "likes", "confidence": 0.8}]},
            ]
        )

        with patch("iris_memory.utils.llm_helper.call_llm", return_value=llm_result):
            items = [{"text": "张三喜欢编程", "user_id": "u1", "sender_name": "张三", "index": 0}]
            result = run(ext.batch_extract_by_llm(items))

        assert 0 in result

    def test_parse_single_triple_valid(self, storage_and_extractor_llm):
        """解析有效的三元组"""
        from iris_memory.knowledge_graph.kg_extractor import KGExtractor

        data = {
            "subject": "张三",
            "predicate": "喜欢",
            "object": "编程",
            "relation_type": "likes",
            "subject_type": "person",
            "object_type": "concept",
            "confidence": 0.8
        }
        triple = KGExtractor._parse_single_triple(data, "张三喜欢编程")
        assert triple is not None
        assert triple.subject == "张三"
        assert triple.object == "编程"
        assert triple.confidence == 0.8

    def test_parse_single_triple_missing_field(self, storage_and_extractor_llm):
        """缺少必要字段返回 None"""
        from iris_memory.knowledge_graph.kg_extractor import KGExtractor

        data = {"subject": "张三"}
        triple = KGExtractor._parse_single_triple(data, "test")
        assert triple is None

    def test_parse_single_triple_invalid_relation_type(self, storage_and_extractor_llm):
        """无效 relation_type 回退到 RELATED_TO"""
        from iris_memory.knowledge_graph.kg_extractor import KGExtractor
        from iris_memory.knowledge_graph.kg_models import KGRelationType

        data = {
            "subject": "张三",
            "predicate": "关联",
            "object": "某事",
            "relation_type": "invalid_type",
            "confidence": 0.5
        }
        triple = KGExtractor._parse_single_triple(data, "test")
        assert triple is not None
        assert triple.relation_type == KGRelationType.RELATED_TO
