"""记忆管理路由 /api/v1/memories"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from quart import request

from iris_memory.web.helpers import safe_int
from iris_memory.web.response import error_response, success_response

logger = logging.getLogger("iris_memory.web")

if TYPE_CHECKING:
    from quart import Quart

    from iris_memory.web.container import WebContainer


def register_memory_routes(app: Quart, container: WebContainer) -> None:

    @app.route("/api/v1/memories", methods=["GET"])
    async def api_list_memories():
        svc = container.get("memory_service")
        data = await svc.search_memories_web(
            query=request.args.get("query", ""),
            user_id=request.args.get("user_id"),
            group_id=request.args.get("group_id"),
            storage_layer=request.args.get("storage_layer"),
            memory_type=request.args.get("type"),
            persona_id=request.args.get("persona_id"),
            page=safe_int(request.args.get("page"), 1),
            page_size=safe_int(request.args.get("page_size"), 20, max_val=100),
        )
        return success_response(data)

    @app.route("/api/v1/memories", methods=["POST"])
    async def api_create_memory():
        data = await request.get_json(silent=True) or {}
        content = data.get("content", "")
        user_id = data.get("user_id", "")
        if not content or not user_id:
            return error_response("content 和 user_id 为必填项")
        svc = container.get("memory_service")
        ok, msg, memory_id = await svc.create_memory_manual(
            content=content,
            user_id=user_id,
            group_id=data.get("group_id"),
            sender_name=data.get("sender_name"),
            persona_id=data.get("persona_id", "default"),
            memory_type=data.get("type", "episodic"),
            storage_layer=data.get("storage_layer", "episodic"),
        )
        if not ok:
            return error_response(msg)
        return success_response({"id": memory_id}, message=msg)

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

    @app.route("/api/v1/memories/<memory_id>/persona", methods=["PATCH"])
    async def api_update_memory_persona(memory_id: str):
        data = await request.get_json(silent=True) or {}
        new_persona_id = data.get("persona_id", "")
        if not new_persona_id:
            return error_response("persona_id 不能为空")
        svc = container.get("memory_service")
        ok, msg = await svc.update_persona_for_memory(memory_id, new_persona_id)
        if not ok:
            return error_response(msg)
        return success_response(message=msg)

    @app.route("/api/v1/bot-personas", methods=["GET"])
    async def api_list_bot_personas():
        persona_set: set[str] = {"default"}
        # Use AstrBot PersonaManager as the authoritative source
        try:
            astrbot_ctx = container._memory_service.context
            pm_personas = astrbot_ctx.persona_manager.personas
            logger.info(
                "[persona] persona_manager.personas count=%d ids=%s",
                len(pm_personas),
                [p.persona_id for p in pm_personas],
            )
            for p in pm_personas:
                persona_set.add(p.persona_id)
        except Exception as e:
            logger.warning("[persona] persona_manager lookup failed: %s", e)
        # Also collect from KG and Chroma metadata for backward compatibility
        try:
            kg_svc = container.get("kg_service")
            for pid in await kg_svc.list_personas():
                persona_set.add(pid)
        except Exception:
            pass
        try:
            mem_svc = container.get("memory_service")
            chroma = mem_svc._service.chroma_manager
            if chroma and chroma.is_ready:
                res = chroma.collection.get(include=["metadatas"])
                for meta in res.get("metadatas", []):
                    pid = meta.get("persona_id")
                    if pid:
                        persona_set.add(pid)
        except Exception:
            pass
        personas = ["default"] + sorted(p for p in persona_set if p != "default")
        logger.info("[persona] bot-personas response: %s", personas)
        return success_response({"personas": personas})

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
