"""Web 知识图谱管理服务"""

from __future__ import annotations

from typing import Any

from iris_memory.utils.logger import get_logger
from iris_memory.web.audit import audit_log
from iris_memory.web.dto.converters import edge_to_web_dict, node_to_web_dict

logger = get_logger("web.kg_svc")


class KgWebService:
    """Web 端知识图谱管理服务"""

    def __init__(self, memory_service: Any) -> None:
        self._service = memory_service
        # Repo lazily created since it needs memory_service
        self._repo: Any = None

    def _get_repo(self) -> Any:
        if self._repo is None:
            from iris_memory.web.repositories.kg_repo import KnowledgeGraphRepository

            self._repo = KnowledgeGraphRepository(self._service)
        return self._repo

    async def search_kg_nodes(
        self,
        query: str = "",
        user_id: str | None = None,
        group_id: str | None = None,
        node_type: str | None = None,
        persona_id: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        repo = self._get_repo()
        nodes, total = await repo.search_nodes(
            query=query,
            user_id=user_id,
            group_id=group_id,
            node_type=node_type,
            persona_id=persona_id,
            page=page,
            page_size=page_size,
        )
        return {"items": [node_to_web_dict(n) for n in nodes], "total": total}

    async def list_kg_edges(
        self,
        user_id: str | None = None,
        group_id: str | None = None,
        relation_type: str | None = None,
        node_id: str | None = None,
        persona_id: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        repo = self._get_repo()
        edges, node_names, total = await repo.list_edges(
            user_id=user_id,
            group_id=group_id,
            relation_type=relation_type,
            node_id=node_id,
            persona_id=persona_id,
            page=page,
            page_size=page_size,
        )
        return {
            "items": [edge_to_web_dict(e, node_names) for e in edges],
            "total": total,
        }

    async def delete_kg_node(self, node_id: str) -> tuple[bool, str]:
        repo = self._get_repo()
        success, msg = await repo.delete_node(node_id)
        if success:
            audit_log("delete_kg_node", f"id={node_id}")
        return success, msg

    async def delete_kg_edge(self, edge_id: str) -> tuple[bool, str]:
        repo = self._get_repo()
        success, msg = await repo.delete_edge(edge_id)
        if success:
            audit_log("delete_kg_edge", f"id={edge_id}")
        return success, msg

    async def update_persona_for_node(
        self, node_id: str, new_persona_id: str
    ) -> tuple[bool, str]:
        """Update the persona_id of a KG node."""
        if not new_persona_id or not new_persona_id.strip():
            return False, "persona_id 不能为空"
        repo = self._get_repo()
        success, msg = await repo.update_node_persona(node_id, new_persona_id.strip())
        if success:
            audit_log("update_kg_node_persona", f"id={node_id} persona={new_persona_id}")
        return success, msg

    async def update_persona_for_edge(
        self, edge_id: str, new_persona_id: str
    ) -> tuple[bool, str]:
        """Update the persona_id of a KG edge."""
        if not new_persona_id or not new_persona_id.strip():
            return False, "persona_id 不能为空"
        repo = self._get_repo()
        success, msg = await repo.update_edge_persona(edge_id, new_persona_id.strip())
        if success:
            audit_log("update_kg_edge_persona", f"id={edge_id} persona={new_persona_id}")
        return success, msg

    async def list_personas(self) -> list[str]:
        """List all distinct persona_id values across KG nodes and edges."""
        return await self._get_repo().list_personas()

    async def get_kg_graph_data(
        self,
        user_id: str | None = None,
        group_id: str | None = None,
        center_node_id: str | None = None,
        depth: int = 2,
        max_nodes: int = 100,
    ) -> dict[str, Any]:
        repo = self._get_repo()
        return await repo.get_graph_data(
            user_id=user_id,
            group_id=group_id,
            center_node_id=center_node_id,
            depth=depth,
            max_nodes=max_nodes,
        )

    async def run_maintenance(self) -> dict[str, Any]:
        repo = self._get_repo()
        result = await repo.run_maintenance()
        if "error" not in result:
            audit_log("kg_maintenance", "completed")
        return result

    async def check_consistency(self) -> dict[str, Any]:
        return await self._get_repo().check_consistency()

    async def get_quality_report(self) -> dict[str, Any]:
        return await self._get_repo().get_quality_report()
