"""系统状态路由 /api/v1/system"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quart import request

from iris_memory.web.response import error_response, success_response

if TYPE_CHECKING:
    from quart import Quart

    from iris_memory.web.container import WebContainer


def register_system_routes(app: Quart, container: WebContainer) -> None:

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

    @app.route("/api/v1/system/reset", methods=["POST"])
    async def api_system_reset():
        data = await request.get_json(silent=True) or {}
        if not data.get("confirm"):
            return error_response("请确认重置操作")

        scope = (data.get("scope") or "all").strip().lower()
        if scope not in {"all", "kv", "db"}:
            return error_response("scope 仅支持 all / kv / db")

        svc = container.get("system_service")
        result = await svc.reset_data(scope)
        return success_response(result, message="重置完成")
