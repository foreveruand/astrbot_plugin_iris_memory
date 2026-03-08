"""用户画像路由 /api/v1/personas"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quart import request

from iris_memory.web.helpers import safe_int
from iris_memory.web.response import error_response, success_response

if TYPE_CHECKING:
    from quart import Quart
    from iris_memory.web.container import WebContainer


def register_persona_routes(app: "Quart", container: "WebContainer") -> None:

    @app.route("/api/v1/personas", methods=["GET"])
    async def api_list_personas():
        svc = container.get("persona_service")
        data = await svc.list_personas(
            query=request.args.get("query", ""),
            page=safe_int(request.args.get("page"), 1),
            page_size=safe_int(request.args.get("page_size"), 20, max_val=100),
        )
        return success_response(data)

    @app.route("/api/v1/personas/<user_id>", methods=["GET"])
    async def api_get_persona(user_id: str):
        svc = container.get("persona_service")
        detail = await svc.get_persona_detail(user_id)
        if detail is None:
            return error_response("用户画像未找到", 404)
        return success_response(detail)

    @app.route("/api/v1/personas/<user_id>", methods=["DELETE"])
    async def api_delete_persona(user_id: str):
        svc = container.get("persona_service")
        result = await svc.delete_persona(user_id)
        if not result.get("success"):
            return error_response(result.get("message", "删除失败"))
        return success_response(result)

    @app.route("/api/v1/personas/clear", methods=["POST"])
    async def api_clear_personas():
        svc = container.get("persona_service")
        result = await svc.clear_all_personas()
        return success_response(result)

    @app.route("/api/v1/emotions", methods=["GET"])
    async def api_get_emotions():
        svc = container.get("persona_service")
        data = await svc.get_emotion_state(
            user_id=request.args.get("user_id"),
            group_id=request.args.get("group_id"),
        )
        if data is None:
            return error_response("情绪状态未找到", 404)
        return success_response(data)
