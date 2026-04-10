"""系统状态 Web 服务"""

from __future__ import annotations

import os
import time
from typing import Any

from iris_memory.core.constants import KVStoreKeys
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
        result: dict[str, Any] = {"chroma": None, "kg": None, "kv": None}

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

        result["kv"] = {
            "managed_keys": [
                KVStoreKeys.SESSIONS,
                KVStoreKeys.LIFECYCLE_STATE,
                KVStoreKeys.BATCH_QUEUES,
                KVStoreKeys.CHAT_HISTORY,
                KVStoreKeys.USER_PERSONAS,
                KVStoreKeys.MEMBER_IDENTITY,
                KVStoreKeys.GROUP_ACTIVITY,
                KVStoreKeys.PROACTIVE_REPLY_WHITELIST,
                KVStoreKeys.PERSONA_BATCH_QUEUES,
            ],
            "count": 9,
        }

        return result

    def get_overview(self) -> dict[str, Any]:
        """系统概览（仪表盘顶部）"""
        return {
            "health": self.health(),
            "storage": self.get_storage_stats(),
            "pid": os.getpid(),
        }

    async def reset_data(self, scope: str = "all") -> dict[str, Any]:
        """从 WebUI 重置 Iris Memory 数据。"""
        keys_to_delete = [
            KVStoreKeys.SESSIONS,
            KVStoreKeys.LIFECYCLE_STATE,
            KVStoreKeys.BATCH_QUEUES,
            KVStoreKeys.CHAT_HISTORY,
            KVStoreKeys.USER_PERSONAS,
            KVStoreKeys.MEMBER_IDENTITY,
            KVStoreKeys.GROUP_ACTIVITY,
            KVStoreKeys.PROACTIVE_REPLY_WHITELIST,
            KVStoreKeys.PERSONA_BATCH_QUEUES,
        ]

        deleted_kv_count = 0
        db_deleted_count = 0
        errors: list[str] = []

        if scope in {"all", "kv"}:
            delete_kv_func = getattr(self._service, "_delete_kv_data", None)
            if not delete_kv_func:
                errors.append("delete_kv_data 接口不可用")
            else:
                for key in keys_to_delete:
                    try:
                        await delete_kv_func(key)
                        deleted_kv_count += 1
                    except Exception as e:
                        errors.append(f"{key}: {e}")
                        logger.warning(f"Failed to delete KV key {key}: {e}")

            try:
                self._service._user_personas.clear()
                self._service._user_emotional_states.clear()
                self._service._recently_injected.clear()
            except Exception as e:
                errors.append(f"clear_runtime_cache: {e}")

        if scope in {"all", "db"}:
            try:
                success, db_deleted_count = await self._service.delete_all_memories()
                if not success:
                    errors.append("delete_all_memories returned False")
            except Exception as e:
                errors.append(f"chroma_memories: {e}")
                logger.warning(f"Failed to delete all memories from WebUI: {e}")

        return {
            "scope": scope,
            "deleted_kv_count": deleted_kv_count,
            "total_kv_keys": len(keys_to_delete),
            "db_deleted_count": db_deleted_count,
            "errors": errors,
        }
