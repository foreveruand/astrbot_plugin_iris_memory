"""
事件发射器 — 配置变更 Pub/Sub 广播

当 WebUI 触发配置修改时，仅通知订阅了该键（或 section）的模块执行热重载，
避免整个插件重启。

使用方式::

    from iris_memory.config.events import config_events

    # 订阅特定配置键变更
    config_events.on("proactive_reply.cooldown_seconds", my_handler)

    # 订阅某个 section 的任何变更
    config_events.on_section("proactive_reply", my_section_handler)

    # 订阅所有配置变更
    config_events.on_any(my_global_handler)

    # 触发变更通知（由 ConfigStore 内部调用）
    config_events.emit("proactive_reply.cooldown_seconds", old_val, new_val)
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# 回调签名: (key: str, old_value: Any, new_value: Any) -> None
ChangeHandler = Callable[[str, Any, Any], None]
AsyncChangeHandler = Callable[[str, Any, Any], Any]  # 可以是 async


class ConfigEventEmitter:
    """配置变更事件发射器（线程安全）

    支持三种粒度的订阅：
    1. 精确键：``on("proactive_reply.cooldown_seconds", handler)``
    2. Section：``on_section("proactive_reply", handler)``
    3. 全局：``on_any(handler)``
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key_handlers: dict[str, list[ChangeHandler]] = {}
        self._section_handlers: dict[str, list[ChangeHandler]] = {}
        self._global_handlers: list[ChangeHandler] = []

    # ── 订阅 ──

    def on(self, key: str, handler: ChangeHandler) -> Callable[[], None]:
        """订阅特定配置键的变更

        Returns:
            取消订阅的函数
        """
        with self._lock:
            self._key_handlers.setdefault(key, []).append(handler)

        def unsubscribe() -> None:
            with self._lock:
                handlers = self._key_handlers.get(key)
                if handlers and handler in handlers:
                    handlers.remove(handler)

        return unsubscribe

    def on_section(self, section: str, handler: ChangeHandler) -> Callable[[], None]:
        """订阅某个 section 下任何键的变更

        Returns:
            取消订阅的函数
        """
        with self._lock:
            self._section_handlers.setdefault(section, []).append(handler)

        def unsubscribe() -> None:
            with self._lock:
                handlers = self._section_handlers.get(section)
                if handlers and handler in handlers:
                    handlers.remove(handler)

        return unsubscribe

    def on_any(self, handler: ChangeHandler) -> Callable[[], None]:
        """订阅所有配置变更

        Returns:
            取消订阅的函数
        """
        with self._lock:
            self._global_handlers.append(handler)

        def unsubscribe() -> None:
            with self._lock:
                if handler in self._global_handlers:
                    self._global_handlers.remove(handler)

        return unsubscribe

    # ── 触发 ──

    def emit(self, key: str, old_value: Any, new_value: Any) -> None:
        """触发配置变更事件

        同步通知所有订阅者。如果有 async handler，自动调度到事件循环。
        """
        section = key.split(".")[0] if "." in key else ""

        with self._lock:
            handlers = list(self._key_handlers.get(key, []))
            section_hs = list(self._section_handlers.get(section, []))
            global_hs = list(self._global_handlers)

        for h in handlers + section_hs + global_hs:
            try:
                result = h(key, old_value, new_value)
                # 如果返回的是协程，尝试调度
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        # 没有运行中的事件循环，忽略
                        pass
            except Exception:
                logger.exception("配置变更处理器异常: key=%s, handler=%s", key, h)

    def emit_batch(self, changes: dict[str, tuple]) -> None:
        """批量触发变更事件

        Args:
            changes: ``{key: (old_value, new_value)}``
        """
        for key, (old_val, new_val) in changes.items():
            self.emit(key, old_val, new_val)

    # ── 清理 ──

    def clear(self) -> None:
        """清除所有订阅"""
        with self._lock:
            self._key_handlers.clear()
            self._section_handlers.clear()
            self._global_handlers.clear()


# 全局单例
config_events = ConfigEventEmitter()
