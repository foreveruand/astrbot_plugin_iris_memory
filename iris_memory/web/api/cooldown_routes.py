"""冷却管理路由 /api/v1/cooldown"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quart import request

from iris_memory.web.response import error_response, success_response

if TYPE_CHECKING:
    from quart import Quart

    from iris_memory.web.container import WebContainer


def register_cooldown_routes(app: Quart, container: WebContainer) -> None:

    @app.route("/api/v1/cooldown", methods=["GET"])
    async def api_cooldown_overview():
        svc = container.get("cooldown_service")
        data = svc.get_all_status()
        return success_response(data)

    @app.route("/api/v1/cooldown/<group_id>", methods=["GET"])
    async def api_cooldown_status(group_id: str):
        svc = container.get("cooldown_service")
        data = svc.get_status(group_id)
        return success_response(data)

    @app.route("/api/v1/cooldown/<group_id>/activate", methods=["POST"])
    async def api_cooldown_activate(group_id: str):
        data = await request.get_json(silent=True) or {}
        svc = container.get("cooldown_service")
        result = svc.activate(
            group_id,
            duration_minutes=data.get("duration_minutes"),
            reason=data.get("reason"),
        )
        if not result.get("success"):
            return error_response(result.get("message", "激活失败"))
        return success_response(result)

    @app.route("/api/v1/cooldown/<group_id>/deactivate", methods=["POST"])
    async def api_cooldown_deactivate(group_id: str):
        svc = container.get("cooldown_service")
        result = svc.deactivate(group_id)
        if not result.get("success"):
            return error_response(result.get("message", "停用失败"))
        return success_response(result)
