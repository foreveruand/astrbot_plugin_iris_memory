"""冷却管理 Web 服务"""

from __future__ import annotations

from typing import Any


class CooldownWebService:
    """冷却管理 Web 服务层"""

    def __init__(self, memory_service: Any) -> None:
        self._service = memory_service

    @property
    def _manager(self):
        mod = self._service.cooldown
        return mod.cooldown_manager if mod else None

    def is_available(self) -> bool:
        return self._manager is not None

    def get_all_status(self) -> dict[str, Any]:
        mgr = self._manager
        if not mgr:
            return {"available": False, "active_count": 0, "groups": []}

        return {
            "available": True,
            "active_count": mgr.active_count,
            "default_duration": mgr.default_duration,
        }

    def get_status(self, group_id: str) -> dict[str, Any]:
        mgr = self._manager
        if not mgr:
            return {"active": False, "available": False}

        state = mgr.get_status(group_id)
        if not state:
            return {"active": False, "group_id": group_id}

        return {
            "active": state.is_active,
            "group_id": group_id,
            "reason": state.reason,
            "initiated_by": state.initiated_by,
            "remaining_seconds": state.remaining_seconds,
            "formatted": mgr.format_status(group_id),
        }

    def activate(
        self,
        group_id: str,
        duration_minutes: int | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        mgr = self._manager
        if not mgr:
            return {"success": False, "message": "冷却管理器未就绪"}

        msg = mgr.activate(
            group_id,
            duration_minutes=duration_minutes,
            reason=reason,
            initiated_by="webui",
        )
        return {"success": True, "message": msg}

    def deactivate(self, group_id: str) -> dict[str, Any]:
        mgr = self._manager
        if not mgr:
            return {"success": False, "message": "冷却管理器未就绪"}

        msg = mgr.deactivate(group_id)
        return {"success": True, "message": msg}
