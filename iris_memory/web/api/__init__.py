"""API 路由注册"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quart import Quart

    from iris_memory.web.container import WebContainer


def register_all_routes(app: Quart, container: WebContainer) -> None:
    """注册所有 API 路由到 Quart 应用"""
    from iris_memory.web.api.auth_routes import register_auth_routes
    from iris_memory.web.api.config_routes import register_config_routes
    from iris_memory.web.api.cooldown_routes import register_cooldown_routes
    from iris_memory.web.api.dashboard_routes import register_dashboard_routes
    from iris_memory.web.api.io_routes import register_io_routes
    from iris_memory.web.api.kg_routes import register_kg_routes
    from iris_memory.web.api.llm_routes import register_llm_routes
    from iris_memory.web.api.memory_routes import register_memory_routes
    from iris_memory.web.api.persona_routes import register_persona_routes
    from iris_memory.web.api.proactive_routes import register_proactive_routes
    from iris_memory.web.api.system_routes import register_system_routes

    register_auth_routes(app)
    register_dashboard_routes(app, container)
    register_memory_routes(app, container)
    register_kg_routes(app, container)
    register_persona_routes(app, container)
    register_proactive_routes(app, container)
    register_io_routes(app, container)
    register_cooldown_routes(app, container)
    register_llm_routes(app, container)
    register_config_routes(app, container)
    register_system_routes(app, container)
