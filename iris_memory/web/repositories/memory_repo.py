"""记忆数据仓库

封装 ChromaDB 和 SessionManager 的数据访问。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from iris_memory.core.types import MemoryType, StorageLayer
from iris_memory.models.memory import Memory
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
        group_id: Optional[str] = None,
        storage_layer: Optional[str] = None,
        memory_type: Optional[str] = None,
        top_k: int = 100,
    ) -> List[Dict[str, Any]]:
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
                items.append(item)
            return items
        except Exception as e:
            logger.warning(f"Memory search error: {e}")
            return []

    async def get_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
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
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        storage_layer: Optional[str] = None,
        memory_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """分页列出记忆"""
        page = max(1, page)
        page_size = max(1, min(page_size, 100))
        result: Dict[str, Any] = {"items": [], "total": 0, "page": page, "page_size": page_size}

        all_items: List[Dict[str, Any]] = []

        # 工作记忆（内存缓存）
        if not storage_layer or storage_layer == "working":
            all_items.extend(self._collect_working_memories(user_id, group_id, memory_type))

        # 持久化记忆（ChromaDB）
        if not storage_layer or storage_layer != "working":
            all_items.extend(
                await self._collect_persistent_memories(user_id, group_id, storage_layer, memory_type)
            )

        all_items.sort(key=lambda x: x.get("created_time", ""), reverse=True)
        result["total"] = len(all_items)
        start = (page - 1) * page_size
        result["items"] = all_items[start : start + page_size]

        return result

    async def update(self, memory_id: str, updates: Dict[str, Any]) -> Tuple[bool, str]:
        """更新记忆"""
        allowed_keys = {"content", "type", "storage_layer", "confidence", "importance_score", "summary"}
        invalid_keys = set(updates.keys()) - allowed_keys
        if invalid_keys:
            return False, f"不允许更新的字段: {', '.join(invalid_keys)}"

        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return False, "存储服务未就绪"

            collection = chroma.collection
            res = collection.get(ids=[memory_id], include=["documents", "metadatas"])
            if not res["ids"]:
                return False, "记忆不存在"

            meta = dict(res["metadatas"][0]) if res["metadatas"] else {}
            doc = res["documents"][0] if res.get("documents") else ""

            if "content" in updates:
                doc = updates.pop("content")
            meta.update(updates)

            collection.update(ids=[memory_id], documents=[doc], metadatas=[meta])
            return True, "更新成功"

        except Exception as e:
            logger.error(f"Update memory error: {e}")
            return False, f"更新失败: {e}"

    async def delete(self, memory_id: str) -> Tuple[bool, str]:
        """删除记忆"""
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

    async def batch_delete(self, memory_ids: List[str]) -> Dict[str, Any]:
        """批量删除"""
        result: Dict[str, Any] = {"success_count": 0, "fail_count": 0, "errors": []}

        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                result["errors"].append("存储服务未就绪")
                result["fail_count"] = len(memory_ids)
                return result

            collection = chroma.collection
            res = collection.get(ids=memory_ids)
            existing_ids = set(res["ids"]) if res["ids"] else set()

            if existing_ids:
                collection.delete(ids=list(existing_ids))
                result["success_count"] = len(existing_ids)

            missing = set(memory_ids) - existing_ids
            result["fail_count"] = len(missing)
            if missing:
                result["errors"].append(f"{len(missing)} 条记忆不存在")

        except Exception as e:
            logger.error(f"Batch delete error: {e}")
            result["errors"].append(str(e))
            result["fail_count"] = len(memory_ids)

        return result

    async def count_by_layer(self) -> Dict[str, int]:
        """按层级统计记忆数量"""
        counts: Dict[str, int] = {"working": 0, "episodic": 0, "semantic": 0}

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

    async def get_trend(self, days: int = 30) -> List[Dict[str, Any]]:
        """获取创建趋势"""
        trend: Dict[str, int] = {}
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
        user_id: Optional[str],
        group_id: Optional[str],
        memory_type: Optional[str],
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
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
        user_id: Optional[str],
        group_id: Optional[str],
        storage_layer: Optional[str],
        memory_type: Optional[str],
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return items

            collection = chroma.collection
            where_clause: Dict[str, Any] = {}

            if user_id:
                where_clause["user_id"] = user_id
            if group_id:
                where_clause["group_id"] = group_id
            if storage_layer:
                where_clause["storage_layer"] = storage_layer
            if memory_type:
                where_clause["type"] = memory_type

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
    def _memory_to_dict(memory: Any) -> Dict[str, Any]:
        from iris_memory.web.dto.converters import memory_to_web_dict
        return memory_to_web_dict(memory)

    @staticmethod
    def _dict_from_chroma(res: Dict[str, Any], index: int) -> Dict[str, Any]:
        from iris_memory.web.dto.converters import memory_detail_from_chroma
        return memory_detail_from_chroma(res, index)
