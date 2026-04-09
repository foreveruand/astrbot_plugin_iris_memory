"""
iris_memory.config — 统一配置系统

架构：
  Schema (schema.py)      : 单一数据源，定义全部配置项
  Validators (validators.py): 运行时校验与类型转换
  Loader (loader.py)      : 多层配置源加载与合并
  Store (store.py)        : 核心存储，扁平化访问 API
  Events (events.py)      : Pub/Sub 配置变更广播
  Backup (backup.py)      : 写入前自动备份

使用方式::

    from iris_memory.config import get_store, init_store

    # 初始化（插件启动时调用一次）
    store = init_store(user_config, plugin_data_path)

    # 读取配置
    store.get("basic.enable_memory")           # True
    store.get("memory.max_context_memories")    # 3

    # 修改可写配置 (Level 2)
    store.set("proactive_reply.cooldown_seconds", 30)

    # 订阅变更
    store.on("proactive_reply.cooldown_seconds", my_handler)

    # WebUI 全量读取
    all_config = store.get_all_for_webui()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from iris_memory.config.backup import ConfigBackup
from iris_memory.config.events import ConfigEventEmitter, config_events
from iris_memory.config.loader import ConfigLoader
from iris_memory.config.schema import (
    ALIAS_MAP,
    SCHEMA,
    AccessLevel,
    ConfigField,
    get_defaults_dict,
    get_section_defaults,
)
from iris_memory.config.store import ConfigStore
from iris_memory.config.validators import (
    ConfigValidationError,
    inject_defaults,
    validate_dict,
    validate_field,
)

__all__ = [
    # 核心
    "ConfigStore",
    "get_store",
    "init_store",
    "reset_store",
    # Schema
    "SCHEMA",
    "ALIAS_MAP",
    "AccessLevel",
    "ConfigField",
    "get_defaults_dict",
    "get_section_defaults",
    # 事件
    "ConfigEventEmitter",
    "config_events",
    # 校验
    "ConfigValidationError",
    "validate_field",
    "validate_dict",
    "inject_defaults",
    # 备份
    "ConfigBackup",
    # 加载
    "ConfigLoader",
]

# ─── 全局 Store 管理 ──────────────────────────────────────

_store: ConfigStore | None = None
_store_lock = __import__("threading").Lock()


def init_store(
    user_config: Any = None,
    plugin_data_path: Path | None = None,
    *,
    cache_ttl: float | None = None,
) -> ConfigStore:
    """初始化全局 ConfigStore（线程安全）

    Args:
        user_config: AstrBot 用户配置对象
        plugin_data_path: 插件数据目录
        cache_ttl: 配置缓存 TTL（秒）

    Returns:
        ConfigStore 实例
    """
    global _store
    with _store_lock:
        _store = ConfigStore(
            user_config=user_config,
            plugin_data_path=plugin_data_path,
            cache_ttl=cache_ttl,
        )
        return _store


def get_store() -> ConfigStore:
    """获取全局 ConfigStore

    如果尚未初始化，创建一个仅包含默认值的实例。
    """
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ConfigStore()
    return _store


def reset_store() -> None:
    """重置全局 ConfigStore（主要用于测试）"""
    global _store
    with _store_lock:
        _store = None
        config_events.clear()
