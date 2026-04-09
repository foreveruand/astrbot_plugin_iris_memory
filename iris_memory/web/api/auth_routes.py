"""认证路由 /api/v1/auth"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quart import request

from iris_memory.web.auth import (
    AuthMiddleware,
    check_login_rate_limit,
    record_login_attempt,
)
from iris_memory.web.response import error_response, success_response

if TYPE_CHECKING:
    from quart import Quart


def register_auth_routes(app: Quart) -> None:
    """注册认证相关路由，auth_middleware 从 app.config 中获取"""

    @app.route("/api/v1/auth/login", methods=["POST"])
    async def api_login():
        auth: AuthMiddleware = app.config["AUTH_MIDDLEWARE"]
        if not auth.require_auth:
            return success_response({"token": None, "message": "无需认证"})

        client_ip = request.remote_addr or "unknown"
        if not check_login_rate_limit(client_ip):
            return error_response("登录尝试过于频繁，请稍后再试", 429)

        record_login_attempt(client_ip)

        data = await request.get_json(silent=True) or {}
        access_key = data.get("access_key", "")

        if not access_key:
            return error_response("请输入访问密钥", 401)

        if not auth.check_auth(request):
            # check_auth 检查 header/query，这里手动比对
            if access_key != app.config.get("ACCESS_KEY", ""):
                return error_response("访问密钥错误", 401)

        token = auth.create_session_token()
        return success_response({"token": token})

    @app.route("/api/v1/auth/check", methods=["GET"])
    async def api_auth_check():
        auth: AuthMiddleware = app.config["AUTH_MIDDLEWARE"]
        if not auth.require_auth:
            return success_response({"authenticated": True, "auth_required": False})

        authenticated = auth.check_auth(request)
        return success_response(
            {
                "authenticated": authenticated,
                "auth_required": True,
            }
        )
