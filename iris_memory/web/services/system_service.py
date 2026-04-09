"""系统状态 Web 服务"""

from __future__ import annotations

import os
import time
from typing import Any

from iris_memory.utils.logger import get_logger

logger = get_logger("web.system_svc")

_START_TIME = time.time()


class SystemWebService:
    """系统健康状态与存储统计"""

    def __init__(self, memory_service: Any) -> None:
        self._service = memory_service

    def health(self) -> dict[str, Any]:
        """返回健康检查信息"""
        return {
            "status": "ok",
            "uptime_seconds": round(time.time() - _START_TIME, 1),
            "initialized": self._service.is_initialized if self._service else False,
        }

    def get_storage_stats(self) -> dict[str, Any]:
        """存储统计"""
        result: dict[str, Any] = {"chroma": None, "kg": None}

        try:
            chroma = self._service.chroma_manager
            if chroma and chroma.is_ready:
                col = chroma.collection
                result["chroma"] = {
                    "ready": True,
                    "count": col.count() if col else 0,
                }
        except Exception as e:
            result["chroma"] = {"ready": False, "error": str(e)}

        try:
            kg = self._service.kg
            if kg and kg.enabled:
                result["kg"] = {"enabled": True}
            else:
                result["kg"] = {"enabled": False}
        except Exception as e:
            result["kg"] = {"enabled": False, "error": str(e)}

        return result

    def get_overview(self) -> dict[str, Any]:
        """系统概览（仪表盘顶部）"""
        return {
            "health": self.health(),
            "storage": self.get_storage_stats(),
            "pid": os.getpid(),
        }
