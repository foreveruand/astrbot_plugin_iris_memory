"""Web 模块 — Iris Memory Web 管理界面

架构层次：
- api/            路由层（按领域拆分）
- services/       业务编排层
- repositories/   数据访问层
- dto/            数据传输对象与转换器

入口：
- WebUIManager    插件侧调用，管理 Web UI 生命周期
- StandaloneWebServer  独立 Quart 应用
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iris_memory.web.server import StandaloneWebServer as StandaloneWebServer
    from iris_memory.web.web_ui import WebUIManager as WebUIManager


def __getattr__(name: str):
    if name == "StandaloneWebServer":
        from iris_memory.web.server import StandaloneWebServer

        return StandaloneWebServer
    if name == "WebUIManager":
        from iris_memory.web.web_ui import WebUIManager

        return WebUIManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "StandaloneWebServer",
    "WebUIManager",
]
