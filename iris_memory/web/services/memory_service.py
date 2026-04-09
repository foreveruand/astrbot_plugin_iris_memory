"""Web 记忆管理服务"""

from __future__ import annotations

from typing import Any

from iris_memory.utils.logger import get_logger
from iris_memory.web.audit import audit_log
from iris_memory.web.dto.converters import memory_detail_from_chroma

logger = get_logger("web.memory_svc")


class MemoryWebService:
    """Web 端记忆管理服务"""

    def __init__(self, memory_service: Any, memory_repo: Any) -> None:
        self._service = memory_service
        self._repo = memory_repo

    async def search_memories_web(
        self,
        query: str = "",
        user_id: str | None = None,
        group_id: str | None = None,
        storage_layer: str | None = None,
        memory_type: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        page = max(1, page)
        page_size = max(1, min(page_size, 100))

        if query and user_id:
            items = await self._repo.search(
                query=query,
                user_id=user_id,
                group_id=group_id,
                storage_layer=storage_layer,
                memory_type=memory_type,
                top_k=100,
            )
            total = len(items)
            start = (page - 1) * page_size
            return {
                "items": items[start : start + page_size],
                "total": total,
                "page": page,
                "page_size": page_size,
            }

        return await self._repo.list_all(
            user_id=user_id,
            group_id=group_id,
            storage_layer=storage_layer,
            memory_type=memory_type,
            page=page,
            page_size=page_size,
        )

    async def get_memory_detail(self, memory_id: str) -> dict[str, Any] | None:
        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return None

            collection = chroma.collection
            res = collection.get(ids=[memory_id], include=["documents", "metadatas"])
            if not res["ids"]:
                return None

            return memory_detail_from_chroma(res, 0, full=True)
        except Exception as e:
            logger.warning(f"Get memory detail error: {e}")
            return None

    async def update_memory_by_id(
        self,
        memory_id: str,
        updates: dict[str, Any],
    ) -> tuple[bool, str]:
        success, msg = await self._repo.update(memory_id, updates)
        if success:
            audit_log("update_memory", f"id={memory_id} fields={list(updates.keys())}")
        return success, msg

    async def delete_memory_by_id(self, memory_id: str) -> tuple[bool, str]:
        success, msg = await self._repo.delete(memory_id)
        if success:
            audit_log("delete_memory", f"id={memory_id}")
        return success, msg

    async def batch_delete_memories(self, memory_ids: list[str]) -> dict[str, Any]:
        result = await self._repo.batch_delete(memory_ids)
        audit_log(
            "batch_delete_memories",
            f"total={len(memory_ids)} success={result['success_count']}",
        )
        return result
