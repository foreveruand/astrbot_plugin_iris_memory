"""
用户隔离聚类测试

验证不同 user_id 的记忆不会被混合聚类。
"""

import pytest
from datetime import datetime, timedelta

from iris_memory.models.memory import Memory
from iris_memory.core.types import MemoryType, StorageLayer
from iris_memory.capture.semantic.semantic_clustering import (
    SemanticClustering,
    MemoryCluster,
)


def _make_memory(
    content: str = "test",
    user_id: str = "user_1",
    keywords: list | None = None,
    days_old: int = 60,
    confidence: float = 0.6,
) -> Memory:
    m = Memory(
        content=content,
        user_id=user_id,
        confidence=confidence,
        storage_layer=StorageLayer.EPISODIC,
        created_time=datetime.now() - timedelta(days=days_old),
    )
    if keywords:
        m.keywords = keywords
    return m


class TestUserIsolation:
    """用户隔离测试"""

    def test_different_users_not_mixed(self):
        """不同用户的记忆不会被聚到同一个 cluster"""
        clustering = SemanticClustering(min_cluster_size=2, cluster_time_window_days=120)
        memories = [
            _make_memory("喜欢火锅", user_id="user_A", keywords=["火锅"], days_old=40),
            _make_memory("吃了火锅", user_id="user_A", keywords=["火锅"], days_old=50),
            _make_memory("喜欢火锅", user_id="user_B", keywords=["火锅"], days_old=40),
            _make_memory("吃了火锅", user_id="user_B", keywords=["火锅"], days_old=50),
        ]
        clusters = clustering.cluster(memories)

        for cluster in clusters:
            user_ids = set(m.user_id for m in cluster.memories)
            assert len(user_ids) == 1, f"Cluster {cluster.cluster_id} contains mixed users: {user_ids}"

    def test_cluster_has_user_id(self):
        """聚类结果的 user_id 字段被正确设置"""
        clustering = SemanticClustering(min_cluster_size=2, cluster_time_window_days=120)
        memories = [
            _make_memory("火锅A", user_id="user_X", keywords=["火锅"], days_old=40),
            _make_memory("火锅B", user_id="user_X", keywords=["火锅"], days_old=50),
        ]
        clusters = clustering.cluster(memories)
        assert len(clusters) >= 1
        assert clusters[0].user_id == "user_X"

    def test_same_user_clustered_together(self):
        """同一用户的相同主题记忆被聚到一起"""
        clustering = SemanticClustering(min_cluster_size=2, cluster_time_window_days=120)
        memories = [
            _make_memory("火锅A", user_id="user_1", keywords=["火锅"], days_old=40),
            _make_memory("火锅B", user_id="user_1", keywords=["火锅"], days_old=50),
            _make_memory("火锅C", user_id="user_1", keywords=["火锅"], days_old=60),
        ]
        clusters = clustering.cluster(memories)
        fire_clusters = [c for c in clusters if c.cluster_key == "火锅"]
        assert len(fire_clusters) == 1
        assert fire_clusters[0].size == 3

    def test_insufficient_per_user_not_clustered(self):
        """单用户记忆不够 min_cluster_size 时不形成 cluster"""
        clustering = SemanticClustering(min_cluster_size=3, cluster_time_window_days=120)
        memories = [
            # user_A 只有 2 条，不够 3
            _make_memory("火锅A", user_id="user_A", keywords=["火锅"], days_old=40),
            _make_memory("火锅B", user_id="user_A", keywords=["火锅"], days_old=50),
            # user_B 也只有 2 条
            _make_memory("火锅C", user_id="user_B", keywords=["火锅"], days_old=40),
            _make_memory("火锅D", user_id="user_B", keywords=["火锅"], days_old=50),
        ]
        clusters = clustering.cluster(memories)
        # 如果不隔离，4条会被聚在一起达到 min_cluster_size=3
        # 隔离后，每个用户只有2条，不够
        assert len(clusters) == 0


class TestGroupByUser:
    """_group_by_user 测试"""

    def test_groups_correctly(self):
        groups = SemanticClustering._group_by_user([
            _make_memory("a", user_id="u1"),
            _make_memory("b", user_id="u2"),
            _make_memory("c", user_id="u1"),
        ])
        assert len(groups) == 2
        assert len(groups["u1"]) == 2
        assert len(groups["u2"]) == 1

    def test_empty_list(self):
        groups = SemanticClustering._group_by_user([])
        assert len(groups) == 0


class TestVectorClusteringUserIsolation:
    """向量聚类的用户隔离测试"""

    def test_vector_clustering_isolates_users(self):
        """cluster_with_vectors 也按用户隔离"""
        import numpy as np
        clustering = SemanticClustering(
            min_cluster_size=2,
            cluster_time_window_days=120,
            similarity_threshold=0.9,
        )
        emb = [1.0, 0.0, 0.0]
        memories = [
            _make_memory("a", user_id="u1", keywords=["独特A"], days_old=40),
            _make_memory("b", user_id="u1", keywords=["独特B"], days_old=50),
            _make_memory("c", user_id="u2", keywords=["独特C"], days_old=40),
            _make_memory("d", user_id="u2", keywords=["独特D"], days_old=50),
        ]
        # Give all memories identical embeddings
        for m in memories:
            m.embedding = np.array(emb)

        clusters = clustering.cluster_with_vectors(memories)

        for cluster in clusters:
            user_ids = set(m.user_id for m in cluster.memories)
            assert len(user_ids) == 1, f"Vector cluster mixed users: {user_ids}"
