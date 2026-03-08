"""会话仓库"""

from __future__ import annotations

from typing import Any, Dict


class SessionRepository:
    """会话仓库"""

    def __init__(self, memory_service: Any) -> None:
        self._service = memory_service

    async def get_all_sessions(self) -> Dict[str, Any]:
        try:
            if self._service.session_manager:
                return self._service.session_manager.get_all_sessions()
            return {}
        except Exception:
            return {}

    async def get_session_stats(self) -> Dict[str, int]:
        result: Dict[str, int] = {
            "total_sessions": 0,
            "active_sessions": 0,
            "total_users": 0,
        }

        try:
            if not self._service.session_manager:
                return result

            sessions = self._service.session_manager.get_all_sessions()
            result["total_sessions"] = len(sessions)

            user_ids = set()
            active = 0
            for key, _meta in sessions.items():
                user_ids.add(key.split(":")[0] if ":" in key else key)
                wm = self._service.session_manager.working_memory_cache.get(key, [])
                if wm:
                    active += 1

            result["active_sessions"] = active
            result["total_users"] = len(user_ids)
        except Exception:
            pass

        return result
