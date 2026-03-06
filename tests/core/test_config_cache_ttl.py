"""配置存储测试 - cache_ttl 可配置"""

import time
import pytest
from iris_memory.config import ConfigStore, reset_store


class TestConfigStoreCacheTTL:
    """ConfigStore cache_ttl 参数化测试"""

    def test_default_cache_ttl(self):
        """默认 TTL 为 10 秒"""
        reset_store()
        store = ConfigStore()
        assert store._cache_ttl == ConfigStore.DEFAULT_CACHE_TTL

    def test_custom_cache_ttl(self):
        """可通过构造参数自定义 TTL"""
        reset_store()
        store = ConfigStore(cache_ttl=60.0)
        assert store._cache_ttl == 60.0

    def test_zero_cache_ttl_always_refetch(self):
        """TTL=0 时每次都重新获取"""
        reset_store()
        store = ConfigStore(cache_ttl=0.0)
        val = store.get("basic.enable_memory", "default_val")
        assert val is not None

    def test_cache_invalidation(self):
        """手动失效缓存"""
        reset_store()
        store = ConfigStore(cache_ttl=3600.0)
        store.get("basic.enable_memory", True)
        assert "basic.enable_memory" in store._cache
        store.invalidate_cache("basic.enable_memory")
        assert "basic.enable_memory" not in store._cache

    def test_cache_invalidation_all(self):
        """失效所有缓存"""
        reset_store()
        store = ConfigStore(cache_ttl=3600.0)
        store.get("basic.enable_memory", True)
        store.get("memory.max_context_memories", 3)
        store.invalidate_cache()
        assert len(store._cache) == 0
