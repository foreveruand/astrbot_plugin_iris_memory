"""记忆管理路由 /api/v1/memories"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quart import request

from iris_memory.web.helpers import safe_int
from iris_memory.web.response import error_response, success_response

if TYPE_CHECKING:
    from quart import Quart
    from iris_memory.web.container import WebContainer


def register_memory_routes(app: "Quart", container: "WebContainer") -> None:

    @app.route("/api/v1/memories", methods=["GET"])
    async def api_list_memories():
        svc = container.get("memory_service")
        data = await svc.search_memories_web(
            query=request.args.get("query", ""),
            user_id=request.args.get("user_id"),
            group_id=request.args.get("group_id"),
            storage_layer=request.args.get("storage_layer"),
            memory_type=request.args.get("type"),
            page=safe_int(request.args.get("page"), 1),
            page_size=safe_int(request.args.get("page_size"), 20, max_val=100),
        )
        return success_response(data)

    @app.route("/api/v1/memories/<memory_id>", methods=["GET"])
    async def api_get_memory(memory_id: str):
        svc = container.get("memory_service")
        detail = await svc.get_memory_detail(memory_id)
        if detail is None:
            return error_response("记忆未找到", 404)
        return success_response(detail)

    @app.route("/api/v1/memories/<memory_id>", methods=["PUT"])
    async def api_update_memory(memory_id: str):
        data = await request.get_json(silent=True) or {}
        if not data:
            return error_response("请求体不能为空")

        svc = container.get("memory_service")
        ok, msg = await svc.update_memory_by_id(memory_id, data)
        if not ok:
            return error_response(msg)
        return success_response(message=msg)

    @app.route("/api/v1/memories/<memory_id>", methods=["DELETE"])
    async def api_delete_memory(memory_id: str):
        svc = container.get("memory_service")
        ok, msg = await svc.delete_memory_by_id(memory_id)
        if not ok:
            return error_response(msg)
        return success_response(message=msg)

    @app.route("/api/v1/memories/batch-delete", methods=["POST"])
    async def api_batch_delete_memories():
        data = await request.get_json(silent=True) or {}
        ids = data.get("ids", [])
        if not ids or not isinstance(ids, list):
            return error_response("请提供要删除的 ID 列表")
        if len(ids) > 500:
            return error_response("单次最多删除 500 条")

        svc = container.get("memory_service")
        result = await svc.batch_delete_memories(ids)
        return success_response(result)
