"""仪表盘路由 /api/v1/dashboard"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quart import request

from iris_memory.web.helpers import safe_int
from iris_memory.web.response import error_response, success_response

if TYPE_CHECKING:
    from quart import Quart
    from iris_memory.web.container import WebContainer


def register_dashboard_routes(app: "Quart", container: "WebContainer") -> None:

    @app.route("/api/v1/dashboard/stats", methods=["GET"])
    async def api_dashboard_stats():
        svc = container.get("dashboard_service")
        data = await svc.get_dashboard_stats()
        return success_response(data)

    @app.route("/api/v1/dashboard/trend", methods=["GET"])
    async def api_dashboard_trend():
        days = safe_int(request.args.get("days"), 30, min_val=1, max_val=365)
        svc = container.get("dashboard_service")
        data = await svc.get_memory_trend(days=days)
        return success_response(data)
