"""主动回复路由 /api/v1/proactive"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quart import request

from iris_memory.web.response import error_response, success_response

if TYPE_CHECKING:
    from quart import Quart
    from iris_memory.web.container import WebContainer


def register_proactive_routes(app: "Quart", container: "WebContainer") -> None:

    @app.route("/api/v1/proactive/status", methods=["GET"])
    async def api_proactive_status():
        svc = container.get("proactive_service")
        data = await svc.get_status()
        return success_response(data)

    @app.route("/api/v1/proactive/stats", methods=["GET"])
    async def api_proactive_stats():
        svc = container.get("proactive_service")
        data = await svc.get_stats()
        return success_response(data)

    @app.route("/api/v1/proactive/followup", methods=["GET"])
    async def api_proactive_followup():
        svc = container.get("proactive_service")
        data = await svc.get_followup_status()
        return success_response(data)

    @app.route("/api/v1/proactive/whitelist", methods=["GET"])
    async def api_proactive_whitelist():
        svc = container.get("proactive_service")
        groups = await svc.list_whitelist()
        return success_response({"groups": groups})

    @app.route("/api/v1/proactive/whitelist", methods=["POST"])
    async def api_proactive_whitelist_add():
        data = await request.get_json(silent=True) or {}
        group_id = data.get("group_id", "").strip()
        if not group_id:
            return error_response("请提供 group_id")

        svc = container.get("proactive_service")
        result = await svc.add_to_whitelist(group_id)
        return success_response(result)

    @app.route("/api/v1/proactive/whitelist/<group_id>", methods=["DELETE"])
    async def api_proactive_whitelist_remove(group_id: str):
        svc = container.get("proactive_service")
        result = await svc.remove_from_whitelist(group_id)
        return success_response(result)

    @app.route("/api/v1/proactive/whitelist/<group_id>/check", methods=["GET"])
    async def api_proactive_whitelist_check(group_id: str):
        svc = container.get("proactive_service")
        result = await svc.check_whitelist(group_id)
        return success_response(result)
