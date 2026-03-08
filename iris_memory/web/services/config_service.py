"""配置管理 Web 服务"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from iris_memory.config import get_store
from iris_memory.web.audit import audit_log
from iris_memory.utils.logger import get_logger

logger = get_logger("web.config_svc")


class ConfigWebService:
    """读写 ConfigStore，供 Web UI 使用"""

    def get_all(self) -> List[Dict[str, Any]]:
        """获取所有配置项（含描述、访问级别等元信息）"""
        store = get_store()
        return store.get_all_for_webui()

    def get_writable_keys(self) -> Set[str]:
        store = get_store()
        return store.get_writable_keys()

    def get_value(self, key: str) -> Any:
        store = get_store()
        return store.get(key)

    def set_value(self, key: str, value: Any) -> Dict[str, Any]:
        store = get_store()
        try:
            store.set(key, value)
            audit_log("config_set", f"{key} = {value!r}")
            return {"success": True, "key": key}
        except (KeyError, ValueError, PermissionError) as e:
            return {"success": False, "error": str(e)}

    def set_batch(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        store = get_store()
        errors = store.set_batch(updates)
        if not errors:
            audit_log("config_set_batch", f"keys={list(updates.keys())}")
        return {"success": not errors, "errors": errors}

    def snapshot(self) -> Dict[str, Any]:
        store = get_store()
        return store.snapshot()

    def diff_from_defaults(self) -> Dict[str, Tuple[Any, Any]]:
        store = get_store()
        return store.diff_from_defaults()
