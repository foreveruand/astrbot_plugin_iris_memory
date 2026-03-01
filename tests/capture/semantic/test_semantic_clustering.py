"""
语义聚类模块测试
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np

from iris_memory.models.memory import Memory
from iris_memory.core.types import MemoryType, StorageLayer, QualityLevel
from iris_memory.capture.semantic.semantic_clustering import (
    SemanticClustering,
    MemoryCluster,
)


def _make_memory(
    content: str = "test",
    memory_type: MemoryType = MemoryType.FACT,
    confidence: float = 0.6,
    days_old: int = 60,
    keywords: list | None = None,
    summarized: bool = False,
    storage_layer: StorageLayer = StorageLayer.EPISODIC,
    embedding=None,
    memory_id: str | None = None,
) -> Memory:
    """创建测试记忆"""
    m = Memory(
        content=content,
        type=memory_type,
        confidence=confidence,
        storage_layer=storage_layer,
        created_time=datetime.now() - timedelta(days=days_old),
        summarized=summarized,
    )
    if keywords:
        m.keywords = keywords
    if embedding is not None:
        m.embedding = np.array(embedding)
    if memory_id:
        m.id = memory_id
    return m


class TestPrefilter:
    """预筛选测试"""

    def test_filters_non_episodic(self):
        """非 EPISODIC 记忆被过滤"""
        clustering = SemanticClustering()
        memories = [
            _make_memory(storage_layer=StorageLayer.WORKING),
            _make_memory(storage_layer=StorageLayer.SEMANTIC),
        ]
        result = clustering._prefilter(memories)
        assert len(result) == 0

    def test_filters_already_summarized(self):
        """已被语义提取的记忆被过滤"""
        clustering = SemanticClustering()
        memories = [_make_memory(summarized=True)]
        result = clustering._prefilter(memories)
        assert len(result) == 0

    def test_filters_low_confidence(self):
        """低置信度记忆被过滤"""
        clustering = SemanticClustering(min_confidence=0.4)
        memories = [_make_memory(confidence=0.3)]
        result = clustering._prefilter(memories)
        assert len(result) == 0

    def test_filters_too_new(self):
        """过新的记忆被过滤"""
        clustering = SemanticClustering(min_age_days=30)
        memories = [_make_memory(days_old=10)]
        result = clustering._prefilter(memories)
        assert len(result) == 0

    def test_passes_valid_memory(self):
        """符合条件的记忆通过"""
        clustering = SemanticClustering()
        memories = [
            _make_memory(confidence=0.6, days_old=60),
        ]
        result = clustering._prefilter(memories)
        assert len(result) == 1


class TestEntityTopicClustering:
    """实体/主题聚类测试"""

    def test_cluster_by_keyword(self):
        """按关键词聚类"""
        clustering = SemanticClustering(min_cluster_size=2, cluster_time_window_days=120)
        memories = [
            _make_memory(content="今天去吃了火锅", keywords=["火锅"], days_old=40),
            _make_memory(content="和朋友去吃了重庆火锅", keywords=["火锅"], days_old=50),
            _make_memory(content="看了电影", keywords=["电影"], days_old=60),
        ]
        clusters = clustering._entity_topic_clustering(memories)
        # "火锅" 应该形成一个聚类
        firespot_clusters = [c for c in clusters if c.cluster_key == "火锅"]
        assert len(firespot_clusters) == 1
        assert firespot_clusters[0].size == 2

    def test_cluster_by_type_fallback(self):
        """未被关键词聚类的记忆按类型聚类"""
        clustering = SemanticClustering(min_cluster_size=2, cluster_time_window_days=120)
        memories = [
            _make_memory(content="x", memory_type=MemoryType.EMOTION, keywords=[], days_old=40),
            _make_memory(content="y", memory_type=MemoryType.EMOTION, keywords=[], days_old=50),
            _make_memory(content="z", memory_type=MemoryType.EMOTION, keywords=[], days_old=60),
        ]
        clusters = clustering._entity_topic_clustering(memories)
        type_clusters = [c for c in clusters if c.cluster_type == "type"]
        assert len(type_clusters) >= 1

    def test_time_window_filtering(self):
        """超出时间窗口的记忆不被聚类"""
        clustering = SemanticClustering(
            min_cluster_size=2,
            cluster_time_window_days=30,
        )
        memories = [
            _make_memory(content="火锅好吃", keywords=["火锅"], days_old=40),
            _make_memory(content="又吃火锅", keywords=["火锅"], days_old=50),
        ]
        clusters = clustering._entity_topic_clustering(memories)
        firespot_clusters = [c for c in clusters if c.cluster_key == "火锅"]
        assert len(firespot_clusters) == 0

    def test_min_cluster_size(self):
        """不满足最小聚类大小的组被过滤"""
        clustering = SemanticClustering(min_cluster_size=3, cluster_time_window_days=120)
        memories = [
            _make_memory(content="火锅 A", keywords=["火锅"], days_old=40),
            _make_memory(content="火锅 B", keywords=["火锅"], days_old=50),
        ]
        clusters = clustering._entity_topic_clustering(memories)
        firespot_clusters = [c for c in clusters if c.cluster_key == "火锅"]
        assert len(firespot_clusters) == 0


class TestVectorClustering:
    """向量相似度聚类测试"""

    def test_vector_clustering_basic(self):
        """基本向量聚类"""
        clustering = SemanticClustering(
            min_cluster_size=2,
            similarity_threshold=0.9,
        )
        # 创建两组相似向量
        emb_a1 = [1.0, 0.0, 0.0]
        emb_a2 = [0.99, 0.1, 0.0]  # 与 a1 很相似
        emb_b1 = [0.0, 1.0, 0.0]  # 与 a 组不相似

        memories = [
            _make_memory(content="a1", embedding=emb_a1, days_old=40),
            _make_memory(content="a2", embedding=emb_a2, days_old=50),
            _make_memory(content="b1", embedding=emb_b1, days_old=60),
        ]
        clusters = clustering._vector_clustering(memories)
        # a1 和 a2 应该成一组
        assert len(clusters) >= 1
        cluster_sizes = [c.size for c in clusters]
        assert 2 in cluster_sizes

    def test_vector_clustering_no_embedding(self):
        """无嵌入向量的记忆被跳过"""
        clustering = SemanticClustering(min_cluster_size=2)
        memories = [
            _make_memory(content="a"),
            _make_memory(content="b"),
        ]
        clusters = clustering._vector_clustering(memories)
        assert len(clusters) == 0


class TestClusterMain:
    """主聚类方法测试"""

    def test_cluster_returns_sorted_by_size(self):
        """聚类结果按大小降序排列"""
        clustering = SemanticClustering(
            min_cluster_size=2,
            cluster_time_window_days=120,
        )
        memories = [
            _make_memory(content="火锅A", keywords=["火锅"], days_old=40),
            _make_memory(content="火锅B", keywords=["火锅"], days_old=50),
            _make_memory(content="电影A", keywords=["电影"], days_old=40),
            _make_memory(content="电影B", keywords=["电影"], days_old=50),
            _make_memory(content="电影C", keywords=["电影"], days_old=60),
        ]
        clusters = clustering.cluster(memories)
        if len(clusters) >= 2:
            assert clusters[0].size >= clusters[1].size

    def test_cluster_truncates_max_per_cluster(self):
        """单个聚类的记忆数量被截断"""
        clustering = SemanticClustering(
            min_cluster_size=2,
            max_memories_per_cluster=3,
            cluster_time_window_days=200,
        )
        memories = [
            _make_memory(content=f"火锅{i}", keywords=["火锅"], days_old=40 + i)
            for i in range(10)
        ]
        clusters = clustering.cluster(memories)
        for cluster in clusters:
            assert cluster.size <= 3

    def test_cluster_empty_input(self):
        """空输入返回空列表"""
        clustering = SemanticClustering()
        assert clustering.cluster([]) == []

    def test_cluster_with_vectors(self):
        """带向量聚类的完整流程"""
        clustering = SemanticClustering(
            min_cluster_size=2,
            cluster_time_window_days=120,
            similarity_threshold=0.9,
        )
        emb_a1 = [1.0, 0.0, 0.0]
        emb_a2 = [0.99, 0.1, 0.0]

        memories = [
            _make_memory(content="火锅A", keywords=["火锅"], days_old=40),
            _make_memory(content="火锅B", keywords=["火锅"], days_old=50),
            _make_memory(content="x", keywords=["独特"], days_old=40, embedding=emb_a1),
            _make_memory(content="y", keywords=["另一个"], days_old=50, embedding=emb_a2),
        ]
        clusters = clustering.cluster_with_vectors(memories)
        # 至少有火锅关键词聚类
        assert len(clusters) >= 1


class TestKeywordExtraction:
    """关键词提取测试"""

    def test_uses_existing_keywords(self):
        """优先使用记忆自带的 keywords"""
        m = _make_memory(content="随便什么", keywords=["Python", "编程"])
        result = SemanticClustering._extract_keywords(m)
        assert "python" in result
        assert "编程" in result

    def test_extracts_from_content(self):
        """从 content 中提取关键词"""
        m = _make_memory(content="今天去公司上班了")
        result = SemanticClustering._extract_keywords(m)
        # 应该提取到中文词
        assert len(result) > 0

    def test_filters_stop_words(self):
        """过滤停用词"""
        m = _make_memory(content="我的一个好朋友")
        result = SemanticClustering._extract_keywords(m)
        assert "我" not in result
        assert "的" not in result
