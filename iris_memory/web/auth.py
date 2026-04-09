"""访问密钥认证"""

from __future__ import annotations

import secrets
import time
from collections import defaultdict
from typing import Any

from iris_memory.utils.logger import get_logger

logger = get_logger("web_auth")

_SESSION_TOKENS: dict[str, float] = {}
_TOKEN_EXPIRE_SECONDS = 3600 * 24

_LOGIN_ATTEMPTS: dict[str, list[float]] = defaultdict(list)
_LOGIN_MAX_ATTEMPTS = 5
_LOGIN_WINDOW_SECONDS = 60


def check_login_rate_limit(client_ip: str) -> bool:
    """检查登录限流，True=允许"""
    now = time.time()
    attempts = _LOGIN_ATTEMPTS[client_ip]
    _LOGIN_ATTEMPTS[client_ip] = [
        t for t in attempts if now - t < _LOGIN_WINDOW_SECONDS
    ]
    return len(_LOGIN_ATTEMPTS[client_ip]) < _LOGIN_MAX_ATTEMPTS


def record_login_attempt(client_ip: str) -> None:
    """记录登录尝试"""
    _LOGIN_ATTEMPTS[client_ip].append(time.time())


class AuthMiddleware:
    """访问密钥认证中间件"""

    def __init__(self, access_key: str) -> None:
        self._access_key = access_key
        self.require_auth = bool(access_key)

    def check_auth(self, request: Any) -> bool:
        """检查请求认证

        支持:
        1. Authorization: Bearer <token>
        2. Query parameter: ?key=<access_key>
        """
        if not self.require_auth:
            return True

        self._cleanup_expired_tokens()

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if token == self._access_key:
                return True
            if token in _SESSION_TOKENS:
                expire_at = _SESSION_TOKENS[token]
                if time.time() < expire_at:
                    return True
                del _SESSION_TOKENS[token]

        query_key = request.args.get("key", "")
        if query_key and query_key == self._access_key:
            return True

        return False

    def create_session_token(self) -> str:
        """创建会话令牌"""
        token = secrets.token_urlsafe(32)
        _SESSION_TOKENS[token] = time.time() + _TOKEN_EXPIRE_SECONDS
        return token

    @staticmethod
    def _cleanup_expired_tokens() -> None:
        now = time.time()
        expired = [t for t, exp in _SESSION_TOKENS.items() if now >= exp]
        for t in expired:
            del _SESSION_TOKENS[t]
