"""配置管理路由 /api/v1/config"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quart import request

from iris_memory.web.response import error_response, success_response

if TYPE_CHECKING:
    from quart import Quart
    from iris_memory.web.container import WebContainer


def register_config_routes(app: "Quart", container: "WebContainer") -> None:

    @app.route("/api/v1/config", methods=["GET"])
    async def api_config_list():
        svc = container.get("config_service")
        data = svc.get_all()
        return success_response(data)

    @app.route("/api/v1/config/<path:key>", methods=["GET"])
    async def api_config_get(key: str):
        svc = container.get("config_service")
        value = svc.get_value(key)
        return success_response({"key": key, "value": value})

    @app.route("/api/v1/config/<path:key>", methods=["PUT"])
    async def api_config_set(key: str):
        data = await request.get_json(silent=True) or {}
        if "value" not in data:
            return error_response("请提供 value 字段")

        svc = container.get("config_service")
        result = svc.set_value(key, data["value"])
        if not result.get("success"):
            return error_response(result.get("error", "设置失败"))
        return success_response(result)

    @app.route("/api/v1/config/batch", methods=["POST"])
    async def api_config_batch():
        data = await request.get_json(silent=True) or {}
        updates = data.get("updates", {})
        if not updates or not isinstance(updates, dict):
            return error_response("请提供 updates 字典")

        svc = container.get("config_service")
        result = svc.set_batch(updates)
        return success_response(result)

    @app.route("/api/v1/config/snapshot", methods=["GET"])
    async def api_config_snapshot():
        svc = container.get("config_service")
        data = svc.snapshot()
        return success_response(data)

    @app.route("/api/v1/config/diff", methods=["GET"])
    async def api_config_diff():
        svc = container.get("config_service")
        data = svc.diff_from_defaults()
        # Convert tuple values to serializable format
        serializable = {k: {"current": v[0], "default": v[1]} for k, v in data.items()}
        return success_response(serializable)
