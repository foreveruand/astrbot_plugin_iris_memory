"""系统状态路由 /api/v1/system"""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris_memory.web.response import success_response

if TYPE_CHECKING:
    from quart import Quart
    from iris_memory.web.container import WebContainer


def register_system_routes(app: "Quart", container: "WebContainer") -> None:

    @app.route("/api/v1/system/health", methods=["GET"])
    async def api_system_health():
        svc = container.get("system_service")
        data = svc.health()
        return success_response(data)

    @app.route("/api/v1/system/storage", methods=["GET"])
    async def api_system_storage():
        svc = container.get("system_service")
        data = svc.get_storage_stats()
        return success_response(data)

    @app.route("/api/v1/system/overview", methods=["GET"])
    async def api_system_overview():
        svc = container.get("system_service")
        data = svc.get_overview()
        return success_response(data)
