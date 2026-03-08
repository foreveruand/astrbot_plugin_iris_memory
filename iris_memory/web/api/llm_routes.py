"""LLM 统计路由 /api/v1/llm"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quart import request

from iris_memory.web.helpers import safe_int
from iris_memory.web.response import success_response

if TYPE_CHECKING:
    from quart import Quart
    from iris_memory.web.container import WebContainer


def register_llm_routes(app: "Quart", container: "WebContainer") -> None:

    @app.route("/api/v1/llm/summary", methods=["GET"])
    async def api_llm_summary():
        svc = container.get("llm_service")
        data = svc.get_summary()
        return success_response(data)

    @app.route("/api/v1/llm/aggregated", methods=["GET"])
    async def api_llm_aggregated():
        svc = container.get("llm_service")
        data = svc.get_aggregated()
        return success_response(data)

    @app.route("/api/v1/llm/recent", methods=["GET"])
    async def api_llm_recent():
        limit = safe_int(request.args.get("limit"), 50, max_val=500)
        svc = container.get("llm_service")
        data = svc.get_recent(limit=limit)
        return success_response(data)

    @app.route("/api/v1/llm/provider/<provider_id>", methods=["GET"])
    async def api_llm_by_provider(provider_id: str):
        svc = container.get("llm_service")
        data = svc.get_by_provider(provider_id)
        return success_response(data)

    @app.route("/api/v1/llm/source/<source>", methods=["GET"])
    async def api_llm_by_source(source: str):
        svc = container.get("llm_service")
        data = svc.get_by_source(source)
        return success_response(data)
