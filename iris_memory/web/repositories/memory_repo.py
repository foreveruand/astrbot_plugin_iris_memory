"""记忆数据仓库

封装 ChromaDB 和 SessionManager 的数据访问。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from iris_memory.core.types import StorageLayer
from iris_memory.utils.logger import get_logger

logger = get_logger("web.memory_repo")


class MemoryRepository:
    """记忆数据仓库"""

    def __init__(self, memory_service: Any) -> None:
        self._service = memory_service

    async def search(
        self,
        query: str,
        user_id: str,
        group_id: str | None = None,
        storage_layer: str | None = None,
        memory_type: str | None = None,
        persona_id: str | None = None,
        top_k: int = 100,
    ) -> list[dict[str, Any]]:
        """向量搜索记忆"""
        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return []

            sl = StorageLayer(storage_layer) if storage_layer else None
            memories = await chroma.query_memories(
                query_text=query,
                user_id=user_id,
                group_id=group_id,
                top_k=top_k,
                storage_layer=sl,
            )

            items = []
            for m in memories:
                item = self._memory_to_dict(m)
                if memory_type and item.get("type") != memory_type:
                    continue
                if persona_id and item.get("persona_id", "default") != persona_id:
                    continue
                items.append(item)
            return items
        except Exception as e:
            logger.warning(f"Memory search error: {e}")
            return []

    async def get_by_id(self, memory_id: str) -> dict[str, Any] | None:
        """根据 ID 获取记忆"""
        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return None

            collection = chroma.collection
            res = collection.get(ids=[memory_id], include=["documents", "metadatas"])

            if not res["ids"]:
                return None

            return self._dict_from_chroma(res, 0)
        except Exception as e:
            logger.warning(f"Get memory by ID error: {e}")
            return None

    async def list_all(
        self,
        user_id: str | None = None,
        group_id: str | None = None,
        storage_layer: str | None = None,
        memory_type: str | None = None,
        persona_id: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """分页列出记忆"""
        page = max(1, page)
        page_size = max(1, min(page_size, 100))
        result: dict[str, Any] = {
            "items": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
        }

        all_items: list[dict[str, Any]] = []

        # 工作记忆（内存缓存）
        if not storage_layer or storage_layer == "working":
            all_items.extend(
                self._collect_working_memories(user_id, group_id, memory_type)
            )

        # 持久化记忆（ChromaDB）
        if not storage_layer or storage_layer != "working":
            all_items.extend(
                await self._collect_persistent_memories(
                    user_id, group_id, storage_layer, memory_type, persona_id
                )
            )

        # 工作记忆按 persona_id 过滤（ChromaDB 不存储工作记忆）
        if persona_id and (not storage_layer or storage_layer == "working"):
            all_items = [
                m
                for m in all_items
                if m.get("persona_id", "default") == persona_id
                or m.get("storage_layer", "") != "working"
            ]

        all_items.sort(key=lambda x: x.get("created_time", ""), reverse=True)
        result["total"] = len(all_items)
        start = (page - 1) * page_size
        result["items"] = all_items[start : start + page_size]

        return result

    async def update(self, memory_id: str, updates: dict[str, Any]) -> tuple[bool, str]:
        """更新记忆"""
        allowed_keys = {
            "content",
            "type",
            "storage_layer",
            "confidence",
            "importance_score",
            "summary",
            "persona_id",
        }
        invalid_keys = set(updates.keys()) - allowed_keys
        if invalid_keys:
            return False, f"不允许更新的字段: {', '.join(invalid_keys)}"

        updates = dict(updates)
        found_in_working = False

        # Update working memory cache if present
        # Note: Memory may exist in both working cache AND ChromaDB if promotion
        # failed to remove from cache. We must update BOTH locations.
        try:
            session_mgr = self._service.session_manager
            if session_mgr and hasattr(session_mgr, "working_memory_cache"):
                from iris_memory.core.types import MemoryType, StorageLayer

                for memories in session_mgr.working_memory_cache.values():
                    for m in memories:
                        if m.id != memory_id:
                            continue
                        found_in_working = True
                        if "content" in updates:
                            m.content = str(updates["content"])
                        if "summary" in updates:
                            m.summary = updates["summary"]
                        if "persona_id" in updates:
                            old_persona = m.persona_id
                            m.persona_id = updates["persona_id"] or "default"
                            logger.info(
                                "[persona] updated working memory id=%s old=%s new=%s",
                                memory_id,
                                old_persona,
                                m.persona_id,
                            )
                        if "confidence" in updates:
                            m.confidence = float(updates["confidence"])
                        if "importance_score" in updates:
                            m.importance_score = float(updates["importance_score"])
                        if "type" in updates:
                            m.type = MemoryType(updates["type"])
                        if "storage_layer" in updates:
                            m.storage_layer = StorageLayer(updates["storage_layer"])
                        # DO NOT return here - continue to check ChromaDB
                        break  # Exit inner loop once found
                    if found_in_working:
                        break  # Exit outer loop once found
        except Exception as e:
            logger.warning(f"Working memory update check error: {e}")

        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                # If found in working cache, return success (working memories don't need Chroma)
                if found_in_working:
                    return True, "更新成功"
                return False, "存储服务未就绪"

            collection = chroma.collection
            res = collection.get(ids=[memory_id], include=["documents", "metadatas"])

            # Memory not in ChromaDB
            if not res["ids"]:
                # If found in working cache, return success (pure working memory)
                if found_in_working:
                    logger.debug(
                        "Memory %s only in working cache, skip ChromaDB update",
                        memory_id,
                    )
                    return True, "更新成功"
                return False, "记忆不存在"

            # Memory exists in ChromaDB - update it
            meta = dict(res["metadatas"][0]) if res["metadatas"] else {}
            doc = res["documents"][0] if res.get("documents") else ""
            old_persona = meta.get("persona_id", "default") or "default"

            if "content" in updates:
                doc = str(updates.pop("content"))
                embedding = await chroma._generate_embedding(doc)
                if embedding is None:
                    return False, "生成嵌入向量失败"
                meta.update(updates)
                collection.update(
                    ids=[memory_id],
                    documents=[doc],
                    embeddings=[embedding],
                    metadatas=[meta],
                )
            else:
                meta.update(updates)
                collection.update(ids=[memory_id], metadatas=[meta])

            logger.info(
                "[persona] updated persistent memory id=%s old=%s new=%s fields=%s chroma_only=%s",
                memory_id,
                old_persona,
                meta.get("persona_id", "default") or "default",
                sorted(updates.keys()),
                not found_in_working,
            )
            return True, "更新成功"

        except Exception as e:
            logger.error(f"Update memory error: {e}")
            # If already updated working cache, consider it partial success
            if found_in_working:
                logger.warning(
                    "Memory %s updated in cache but ChromaDB update failed: %s",
                    memory_id,
                    e,
                )
                return True, "缓存更新成功，持久化存储更新失败"
            return False, f"更新失败: {e}"

    async def delete(self, memory_id: str) -> tuple[bool, str]:
        """删除记忆（同时覆盖工作记忆缓存和 ChromaDB）"""
        # Check working memory cache first (working memories are not in Chroma)
        try:
            session_mgr = self._service.session_manager
            if session_mgr and hasattr(session_mgr, "working_memory_cache"):
                for session_key, memories in session_mgr.working_memory_cache.items():
                    for i, m in enumerate(memories):
                        if m.id == memory_id:
                            session_mgr.working_memory_cache[session_key].pop(i)
                            return True, "删除成功"
        except Exception as e:
            logger.warning(f"Working memory delete check error: {e}")

        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return False, "存储服务未就绪"

            collection = chroma.collection
            res = collection.get(ids=[memory_id])
            if not res["ids"]:
                return False, "记忆不存在"

            collection.delete(ids=[memory_id])
            return True, "删除成功"

        except Exception as e:
            logger.error(f"Delete memory error: {e}")
            return False, f"删除失败: {e}"

    async def batch_delete(self, memory_ids: list[str]) -> dict[str, Any]:
        """批量删除（工作记忆缓存 + ChromaDB）"""
        result: dict[str, Any] = {"success_count": 0, "fail_count": 0, "errors": []}
        remaining_ids = set(memory_ids)

        # Remove working memory entries first
        try:
            session_mgr = self._service.session_manager
            if session_mgr and hasattr(session_mgr, "working_memory_cache"):
                for session_key, memories in list(
                    session_mgr.working_memory_cache.items()
                ):
                    new_list = []
                    for m in memories:
                        if m.id in remaining_ids:
                            remaining_ids.discard(m.id)
                            result["success_count"] += 1
                        else:
                            new_list.append(m)
                    session_mgr.working_memory_cache[session_key] = new_list
        except Exception as e:
            logger.warning(f"Working memory batch delete error: {e}")

        if not remaining_ids:
            return result

        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                result["errors"].append("存储服务未就绪")
                result["fail_count"] += len(remaining_ids)
                return result

            collection = chroma.collection
            res = collection.get(ids=list(remaining_ids))
            existing_ids = set(res["ids"]) if res["ids"] else set()

            if existing_ids:
                collection.delete(ids=list(existing_ids))
                result["success_count"] += len(existing_ids)

            missing = remaining_ids - existing_ids
            result["fail_count"] += len(missing)
            if missing:
                result["errors"].append(f"{len(missing)} 条记忆不存在")

        except Exception as e:
            logger.error(f"Batch delete error: {e}")
            result["errors"].append(str(e))
            result["fail_count"] += len(remaining_ids)

        return result

    async def count_by_layer(self) -> dict[str, int]:
        """按层级统计记忆数量"""
        counts: dict[str, int] = {"working": 0, "episodic": 0, "semantic": 0}

        try:
            session_mgr = self._service.session_manager
            if session_mgr and hasattr(session_mgr, "working_memory_cache"):
                for memories in session_mgr.working_memory_cache.values():
                    counts["working"] += len(memories) if memories else 0
        except Exception:
            pass

        try:
            chroma = self._service.chroma_manager
            if chroma and chroma.is_ready:
                res = chroma.collection.get(include=["metadatas"])
                for meta in res.get("metadatas", []):
                    layer = meta.get("storage_layer", "")
                    if layer in counts:
                        counts[layer] += 1
        except Exception:
            pass

        return counts

    async def get_trend(self, days: int = 30) -> list[dict[str, Any]]:
        """获取创建趋势"""
        trend: dict[str, int] = {}
        now = datetime.now()
        for i in range(days):
            date_str = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            trend[date_str] = 0

        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return [{"date": d, "count": c} for d, c in sorted(trend.items())]

            res = chroma.collection.get(include=["metadatas"])
            for meta in res.get("metadatas", []):
                created = meta.get("created_time", "")
                if created:
                    date_part = created[:10]
                    if date_part in trend:
                        trend[date_part] += 1
        except Exception:
            pass

        return [{"date": d, "count": c} for d, c in sorted(trend.items())]

    # ── Private helpers ──

    def _collect_working_memories(
        self,
        user_id: str | None,
        group_id: str | None,
        memory_type: str | None,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        try:
            session_mgr = self._service.session_manager
            if not session_mgr or not hasattr(session_mgr, "working_memory_cache"):
                return items

            for session_key, memories in session_mgr.working_memory_cache.items():
                key_parts = session_key.split(":")
                session_user_id = key_parts[0] if key_parts else ""
                session_group_id = key_parts[1] if len(key_parts) > 1 else None

                if user_id and session_user_id != user_id:
                    continue
                if group_id and session_group_id != group_id:
                    continue

                for memory in memories or []:
                    item = self._memory_to_dict(memory)
                    if memory_type and item.get("type") != memory_type:
                        continue
                    items.append(item)
        except Exception as e:
            logger.debug(f"Working memory list error: {e}")

        return items

    async def _collect_persistent_memories(
        self,
        user_id: str | None,
        group_id: str | None,
        storage_layer: str | None,
        memory_type: str | None,
        persona_id: str | None = None,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return items

            collection = chroma.collection
            where_clause: dict[str, Any] = {}

            if user_id:
                where_clause["user_id"] = user_id
            if group_id:
                where_clause["group_id"] = group_id
            if storage_layer:
                where_clause["storage_layer"] = storage_layer
            if memory_type:
                where_clause["type"] = memory_type
            if persona_id:
                where_clause["persona_id"] = persona_id

            if where_clause:
                built = chroma._build_where_clause(where_clause)
                res = collection.get(where=built, include=["documents", "metadatas"])
            else:
                res = collection.get(include=["documents", "metadatas"])

            if res["ids"]:
                for i in range(len(res["ids"])):
                    items.append(self._dict_from_chroma(res, i))
        except Exception as e:
            logger.warning(f"List memories error: {e}")

        return items

    @staticmethod
    def _memory_to_dict(memory: Any) -> dict[str, Any]:
        from iris_memory.web.dto.converters import memory_to_web_dict

        return memory_to_web_dict(memory)

    @staticmethod
    def _dict_from_chroma(res: dict[str, Any], index: int) -> dict[str, Any]:
        from iris_memory.web.dto.converters import memory_detail_from_chroma

        return memory_detail_from_chroma(res, index)
