"""仪表盘服务

聚合多维统计数据为仪表盘提供概览。
"""

from __future__ import annotations

from typing import Any

from iris_memory.utils.logger import get_logger

logger = get_logger("web.dashboard_svc")


class DashboardService:
    """仪表盘业务服务"""

    def __init__(
        self, memory_service: Any, memory_repo: Any, session_repo: Any
    ) -> None:
        self._service = memory_service
        self._memory_repo = memory_repo
        self._session_repo = session_repo

    async def get_dashboard_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "system": await self._get_system_stats(),
            "memories": await self._get_memory_overview(),
            "knowledge_graph": await self._get_kg_overview(),
            "health": {},
        }

        try:
            stats["health"] = self._service.health_check()
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            stats["health"] = {"status": "unknown"}

        return stats

    async def get_memory_trend(self, days: int = 30) -> list[dict[str, Any]]:
        return await self._memory_repo.get_trend(days)

    async def _get_system_stats(self) -> dict[str, Any]:
        result = await self._session_repo.get_session_stats()
        try:
            result["total_personas"] = len(self._service._user_personas)
        except Exception:
            result["total_personas"] = 0
        return result

    async def _get_memory_overview(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "total_count": 0,
            "by_layer": {"working": 0, "episodic": 0, "semantic": 0},
            "by_type": {},
        }

        # 工作记忆
        try:
            session_mgr = self._service.session_manager
            if session_mgr and hasattr(session_mgr, "working_memory_cache"):
                for _key, memories in session_mgr.working_memory_cache.items():
                    working_count = len(memories) if memories else 0
                    result["by_layer"]["working"] += working_count
                    result["total_count"] += working_count
                    for mem in memories or []:
                        mtype = getattr(mem, "type", None)
                        if mtype:
                            mtype_val = (
                                mtype.value if hasattr(mtype, "value") else str(mtype)
                            )
                            result["by_type"][mtype_val] = (
                                result["by_type"].get(mtype_val, 0) + 1
                            )
        except Exception as e:
            logger.debug(f"Working memory stats error: {e}")

        # ChromaDB 持久化记忆
        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return result

            collection = chroma.collection
            total = collection.count()
            result["total_count"] += total

            if total == 0:
                return result

            res = collection.get(include=["metadatas"])
            if not res["ids"]:
                return result

            for meta in res["metadatas"]:
                layer = meta.get("storage_layer", "")
                if layer in result["by_layer"]:
                    result["by_layer"][layer] += 1
                mtype = meta.get("type", "")
                if mtype:
                    result["by_type"][mtype] = result["by_type"].get(mtype, 0) + 1

        except Exception as e:
            logger.warning(f"Memory overview error: {e}")

        return result

    async def _get_kg_overview(self) -> dict[str, Any]:
        result: dict[str, Any] = {"nodes": 0, "edges": 0, "enabled": False}
        try:
            kg = self._service.kg
            if kg and kg.enabled:
                result["enabled"] = True
                stats = await kg.get_stats()
                result["nodes"] = stats.get("nodes", 0)
                result["edges"] = stats.get("edges", 0)
        except Exception as e:
            logger.debug(f"KG overview error: {e}")
        return result
