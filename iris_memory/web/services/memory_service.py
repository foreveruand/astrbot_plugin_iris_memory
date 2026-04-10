"""Web 记忆管理服务"""

from __future__ import annotations

import uuid
from datetime import datetime
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
        persona_id: str | None = None,
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
                persona_id=persona_id,
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
            persona_id=persona_id,
            page=page,
            page_size=page_size,
        )

    async def get_memory_detail(self, memory_id: str) -> dict[str, Any] | None:
        try:
            session_mgr = self._service.session_manager
            if session_mgr and hasattr(session_mgr, "working_memory_cache"):
                for memories in session_mgr.working_memory_cache.values():
                    for memory in memories:
                        if memory.id == memory_id:
                            return self._repo._memory_to_dict(memory)

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

    async def update_persona_for_memory(
        self,
        memory_id: str,
        new_persona_id: str,
    ) -> tuple[bool, str]:
        """Update the persona_id of a single memory entry."""
        if not new_persona_id or not new_persona_id.strip():
            return False, "persona_id 不能为空"
        success, msg = await self._repo.update(memory_id, {"persona_id": new_persona_id.strip()})
        if success:
            audit_log("update_memory_persona", f"id={memory_id} persona={new_persona_id}")
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

    async def create_memory_manual(
        self,
        content: str,
        user_id: str,
        group_id: str | None = None,
        sender_name: str | None = None,
        persona_id: str = "default",
        memory_type: str = "episodic",
        storage_layer: str = "episodic",
    ) -> tuple[bool, str, str | None]:
        """Manually add a memory entry via the web UI.

        Returns (success, message, memory_id).
        """
        if not content or not content.strip():
            return False, "内容不能为空", None
        if not user_id or not user_id.strip():
            return False, "user_id 不能为空", None

        try:
            from iris_memory.core.types import MemoryType, StorageLayer
            from iris_memory.models.memory import Memory

            memory = Memory(
                id=str(uuid.uuid4()),
                content=content.strip(),
                summary=content.strip()[:200],
                user_id=user_id.strip(),
                group_id=group_id,
                sender_name=sender_name or user_id.strip(),
                persona_id=persona_id.strip() or "default",
                type=MemoryType(memory_type) if memory_type else MemoryType.EPISODIC,
                storage_layer=StorageLayer(storage_layer) if storage_layer else StorageLayer.EPISODIC,
                created_time=datetime.now(),
                confidence=0.8,
                importance_score=0.5,
            )

            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return False, "存储服务未就绪", None

            memory_id = await chroma.add_memory(memory)
            if memory_id:
                audit_log(
                    "create_memory_manual",
                    f"id={memory_id} user={user_id} persona={persona_id}",
                )
                return True, "记忆创建成功", memory_id
            return False, "存储记忆失败", None

        except Exception as e:
            logger.error(f"Create memory manual error: {e}")
            return False, f"创建失败: {e}", None
