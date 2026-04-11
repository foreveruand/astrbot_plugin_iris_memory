"""知识图谱数据仓库"""

from __future__ import annotations

from typing import Any

from iris_memory.knowledge_graph.kg_models import KGEdge, KGNode, KGNodeType
from iris_memory.utils.logger import get_logger

logger = get_logger("web.kg_repo")


class KnowledgeGraphRepository:
    """知识图谱数据仓库"""

    def __init__(self, memory_service: Any) -> None:
        self._service = memory_service

    def _get_kg(self) -> Any:
        kg = self._service.kg
        if not kg or not kg.enabled:
            return None
        return kg

    async def list_nodes(
        self,
        user_id: str | None = None,
        group_id: str | None = None,
        node_type: str | None = None,
        persona_id: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[KGNode], int]:
        """列出节点，返回 (节点列表, 总数)"""
        kg = self._get_kg()
        if not kg:
            return [], 0

        try:
            storage = kg.storage
            async with storage._lock:
                assert storage._conn

                conditions: list[str] = []
                params: list[Any] = []

                if user_id:
                    conditions.append("user_id = ?")
                    params.append(user_id)
                if group_id:
                    conditions.append("(group_id = ? OR group_id IS NULL)")
                    params.append(group_id)
                if node_type:
                    conditions.append("node_type = ?")
                    params.append(node_type)
                if persona_id:
                    conditions.append("persona_id = ?")
                    params.append(persona_id)

                where_clause = (
                    " WHERE " + " AND ".join(conditions) if conditions else ""
                )

                count_sql = "SELECT COUNT(*) as cnt FROM kg_nodes" + where_clause
                total = storage._conn.execute(count_sql, params).fetchone()["cnt"]

                offset = (page - 1) * page_size
                sql = (
                    "SELECT * FROM kg_nodes"
                    + where_clause
                    + f" ORDER BY created_time DESC LIMIT {int(page_size)} OFFSET {int(offset)}"
                )

                rows = storage._conn.execute(sql, params).fetchall()
                return [KGNode.from_row(dict(r)) for r in rows], total

        except Exception as e:
            logger.warning(f"List KG nodes error: {e}")
            return [], 0

    async def search_nodes(
        self,
        query: str,
        user_id: str | None = None,
        group_id: str | None = None,
        node_type: str | None = None,
        persona_id: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[KGNode], int]:
        """搜索节点，返回 (节点列表, 总数)"""
        kg = self._get_kg()
        if not kg:
            return [], 0

        try:
            if query:
                nt = KGNodeType(node_type) if node_type else None
                all_nodes = await kg.storage.search_nodes(
                    query=query,
                    user_id=user_id,
                    group_id=group_id,
                    node_type=nt,
                    limit=500,
                )
                if persona_id:
                    all_nodes = [
                        n
                        for n in all_nodes
                        if (n.persona_id or "default") == persona_id
                    ]
                total = len(all_nodes)
                offset = (page - 1) * page_size
                return all_nodes[offset : offset + page_size], total
            return await self.list_nodes(
                user_id=user_id,
                group_id=group_id,
                node_type=node_type,
                persona_id=persona_id,
                page=page,
                page_size=page_size,
            )
        except Exception as e:
            logger.warning(f"Search KG nodes error: {e}")
            return [], 0

    async def list_edges(
        self,
        user_id: str | None = None,
        group_id: str | None = None,
        relation_type: str | None = None,
        node_id: str | None = None,
        persona_id: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[KGEdge], dict[str, str], int]:
        """列出边，返回 (边列表, 节点名称映射, 总数)"""
        kg = self._get_kg()
        if not kg:
            return [], {}, 0

        try:
            storage = kg.storage
            async with storage._lock:
                assert storage._conn

                conditions: list[str] = []
                params: list[Any] = []

                if user_id:
                    conditions.append("user_id = ?")
                    params.append(user_id)
                if group_id:
                    conditions.append("(group_id = ? OR group_id IS NULL)")
                    params.append(group_id)
                if relation_type:
                    conditions.append("relation_type = ?")
                    params.append(relation_type)
                if node_id:
                    conditions.append("(source_id = ? OR target_id = ?)")
                    params.extend([node_id, node_id])
                if persona_id:
                    conditions.append("persona_id = ?")
                    params.append(persona_id)

                where_clause = (
                    " WHERE " + " AND ".join(conditions) if conditions else ""
                )

                count_sql = "SELECT COUNT(*) as cnt FROM kg_edges" + where_clause
                total = storage._conn.execute(count_sql, params).fetchone()["cnt"]

                offset = (page - 1) * page_size
                sql = (
                    "SELECT * FROM kg_edges"
                    + where_clause
                    + f" ORDER BY created_time DESC LIMIT {int(page_size)} OFFSET {int(offset)}"
                )

                rows = storage._conn.execute(sql, params).fetchall()
                edges = [KGEdge.from_row(dict(r)) for r in rows]

                node_ids = set()
                for e in edges:
                    node_ids.add(e.source_id)
                    node_ids.add(e.target_id)

                node_names: dict[str, str] = {}
                if node_ids:
                    placeholders = ",".join(["?"] * len(node_ids))
                    nrows = storage._conn.execute(
                        f"SELECT id, display_name, name FROM kg_nodes WHERE id IN ({placeholders})",
                        list(node_ids),
                    ).fetchall()
                    for nr in nrows:
                        nrd = dict(nr)
                        node_names[nrd["id"]] = (
                            nrd.get("display_name") or nrd.get("name") or nrd["id"]
                        )

            return edges, node_names, total

        except Exception as e:
            logger.warning(f"List KG edges error: {e}")
            return [], {}, 0

    async def delete_node(self, node_id: str) -> tuple[bool, str]:
        """删除节点及关联边"""
        kg = self._get_kg()
        if not kg:
            return False, "知识图谱未启用"

        try:
            storage = kg.storage
            async with storage._lock:
                assert storage._conn
                with storage._tx() as cur:
                    cur.execute(
                        "DELETE FROM kg_edges WHERE source_id = ? OR target_id = ?",
                        (node_id, node_id),
                    )
                    edge_count = cur.rowcount
                    cur.execute("DELETE FROM kg_nodes WHERE id = ?", (node_id,))
                    node_count = cur.rowcount

                storage._invalidate_cache()

                if node_count > 0:
                    return True, f"已删除节点及 {edge_count} 条关联边"
                return False, "节点不存在"

        except Exception as e:
            logger.error(f"Delete KG node error: {e}")
            return False, f"删除失败: {e}"

    async def delete_edge(self, edge_id: str) -> tuple[bool, str]:
        """删除边"""
        kg = self._get_kg()
        if not kg:
            return False, "知识图谱未启用"

        try:
            storage = kg.storage
            async with storage._lock:
                assert storage._conn
                with storage._tx() as cur:
                    cur.execute("DELETE FROM kg_edges WHERE id = ?", (edge_id,))
                    if cur.rowcount > 0:
                        return True, "删除成功"
                    return False, "边不存在"

        except Exception as e:
            logger.error(f"Delete KG edge error: {e}")
            return False, f"删除失败: {e}"

    async def get_graph_data(
        self,
        user_id: str | None = None,
        group_id: str | None = None,
        center_node_id: str | None = None,
        depth: int = 2,
        max_nodes: int = 100,
    ) -> dict[str, Any]:
        """获取图谱可视化数据"""
        kg = self._get_kg()
        if not kg:
            return {"nodes": [], "edges": []}

        try:
            from iris_memory.web.dto.converters import (
                edge_to_graph_dict,
                node_to_graph_dict,
            )

            if center_node_id:
                neighbors = await kg.storage.get_neighbors(
                    center_node_id, limit=max_nodes
                )
                node_ids = {center_node_id}
                edges_result = []

                storage = kg.storage
                async with storage._lock:
                    assert storage._conn
                    for n in neighbors:
                        node_ids.add(n.id)
                    if node_ids:
                        placeholders = ",".join(["?"] * len(node_ids))
                        nid_list = list(node_ids)
                        rows = storage._conn.execute(
                            f"SELECT * FROM kg_nodes WHERE id IN ({placeholders})",
                            nid_list,
                        ).fetchall()
                        nodes_result = [KGNode.from_row(dict(r)) for r in rows]

                        erows = storage._conn.execute(
                            f"SELECT * FROM kg_edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                            nid_list + nid_list,
                        ).fetchall()
                        edges_result = [KGEdge.from_row(dict(r)) for r in erows]

                return {
                    "nodes": [node_to_graph_dict(n) for n in nodes_result],
                    "edges": [
                        edge_to_graph_dict(e)
                        for e in edges_result
                        if e.source_id in node_ids and e.target_id in node_ids
                    ],
                }

            # 全局图
            nodes, _ = await self.list_nodes(
                user_id=user_id, group_id=group_id, page=1, page_size=max_nodes
            )
            node_ids_set = {n.id for n in nodes}

            edges_list, _, _ = await self.list_edges(
                user_id=user_id, group_id=group_id, page=1, page_size=max_nodes * 3
            )

            return {
                "nodes": [node_to_graph_dict(n) for n in nodes],
                "edges": [
                    edge_to_graph_dict(e)
                    for e in edges_list
                    if e.source_id in node_ids_set and e.target_id in node_ids_set
                ],
            }

        except Exception as e:
            logger.warning(f"Get KG graph data error: {e}")
            return {"nodes": [], "edges": []}

    async def run_maintenance(self) -> dict[str, Any]:
        """执行知识图谱维护"""
        kg = self._get_kg()
        if not kg:
            return {"error": "知识图谱未启用"}

        try:
            if hasattr(kg, "maintenance") and kg.maintenance:
                result = await kg.maintenance.run_full_maintenance()
                return result
            return {"error": "维护模块未初始化"}
        except Exception as e:
            logger.error(f"KG maintenance error: {e}")
            return {"error": str(e)}

    async def check_consistency(self) -> dict[str, Any]:
        """检查一致性"""
        kg = self._get_kg()
        if not kg:
            return {"error": "知识图谱未启用"}

        try:
            if hasattr(kg, "consistency_checker") and kg.consistency_checker:
                result = await kg.consistency_checker.check()
                return result
            return {"error": "一致性检查模块未初始化"}
        except Exception as e:
            logger.error(f"KG consistency error: {e}")
            return {"error": str(e)}

    async def get_quality_report(self) -> dict[str, Any]:
        """获取质量报告"""
        kg = self._get_kg()
        if not kg:
            return {"error": "知识图谱未启用"}

        try:
            if hasattr(kg, "quality_analyzer") and kg.quality_analyzer:
                result = await kg.quality_analyzer.analyze()
                return result
            return {"error": "质量分析模块未初始化"}
        except Exception as e:
            logger.error(f"KG quality error: {e}")
            return {"error": str(e)}

    async def update_node_persona(
        self, node_id: str, new_persona_id: str
    ) -> tuple[bool, str]:
        """Update the persona_id of a KG node."""
        kg = self._get_kg()
        if not kg:
            return False, "知识图谱未启用"
        try:
            storage = kg.storage
            async with storage._lock:
                assert storage._conn
                with storage._tx() as cur:
                    cur.execute(
                        "UPDATE kg_nodes SET persona_id = ? WHERE id = ?",
                        (new_persona_id, node_id),
                    )
                    if cur.rowcount > 0:
                        storage._invalidate_cache()
                        return True, "节点人格更新成功"
                    return False, "节点不存在"
        except Exception as e:
            logger.error(f"Update node persona error: {e}")
            return False, f"更新失败: {e}"

    async def update_edge_persona(
        self, edge_id: str, new_persona_id: str
    ) -> tuple[bool, str]:
        """Update the persona_id of a KG edge."""
        kg = self._get_kg()
        if not kg:
            return False, "知识图谱未启用"
        try:
            storage = kg.storage
            async with storage._lock:
                assert storage._conn
                with storage._tx() as cur:
                    cur.execute(
                        "UPDATE kg_edges SET persona_id = ? WHERE id = ?",
                        (new_persona_id, edge_id),
                    )
                    if cur.rowcount > 0:
                        storage._invalidate_cache()
                        return True, "边人格更新成功"
                    return False, "边不存在"
        except Exception as e:
            logger.error(f"Update edge persona error: {e}")
            return False, f"更新失败: {e}"

    async def list_personas(self) -> list[str]:
        """List all distinct persona_id values across nodes and edges."""
        kg = self._get_kg()
        if not kg:
            return ["default"]
        try:
            storage = kg.storage
            async with storage._lock:
                assert storage._conn
                node_rows = storage._conn.execute(
                    "SELECT DISTINCT persona_id FROM kg_nodes WHERE persona_id IS NOT NULL"
                ).fetchall()
                edge_rows = storage._conn.execute(
                    "SELECT DISTINCT persona_id FROM kg_edges WHERE persona_id IS NOT NULL"
                ).fetchall()
            personas: set[str] = {"default"}
            for r in node_rows:
                v = dict(r).get("persona_id")
                if v:
                    personas.add(v)
            for r in edge_rows:
                v = dict(r).get("persona_id")
                if v:
                    personas.add(v)
            return sorted(personas)
        except Exception as e:
            logger.warning(f"List personas error: {e}")
            return ["default"]

    async def get_stats(self) -> dict[str, Any]:
        """获取知识图谱统计"""
        kg = self._get_kg()
        if not kg:
            return {"nodes": 0, "edges": 0, "enabled": False}

        try:
            stats = await kg.get_stats()
            return {"enabled": True, **stats}
        except Exception as e:
            logger.debug(f"KG stats error: {e}")
            return {"nodes": 0, "edges": 0, "enabled": True}
