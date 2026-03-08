"""Web 依赖注入容器

管理所有 Web 层服务实例的创建和依赖注入，支持懒加载单例。
"""

from __future__ import annotations

from typing import Any, Callable, Dict


class WebContainer:
    """Web 服务容器

    持有 MemoryService 引用，按需创建 Repository 和 WebService 实例。
    所有实例为单例，在容器生命周期内复用。
    """

    def __init__(self, memory_service: Any) -> None:
        self._memory_service = memory_service
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._register_defaults()

    @property
    def memory_service(self) -> Any:
        return self._memory_service

    def _register_defaults(self) -> None:
        """注册默认服务工厂"""
        # Repositories
        self.register("memory_repo", self._create_memory_repo)
        self.register("kg_repo", self._create_kg_repo)
        self.register("persona_repo", self._create_persona_repo)
        self.register("emotion_repo", self._create_emotion_repo)
        self.register("session_repo", self._create_session_repo)

        # Services
        self.register("dashboard_service", self._create_dashboard_service)
        self.register("memory_service", self._create_memory_web_service)
        self.register("kg_service", self._create_kg_service)
        self.register("persona_service", self._create_persona_service)
        self.register("proactive_service", self._create_proactive_service)
        self.register("io_service", self._create_io_service)
        self.register("cooldown_service", self._create_cooldown_service)
        self.register("llm_service", self._create_llm_service)
        self.register("config_service", self._create_config_service)
        self.register("system_service", self._create_system_service)

    def register(self, name: str, factory: Callable[[], Any]) -> None:
        """注册服务工厂"""
        self._factories[name] = factory

    def get(self, name: str) -> Any:
        """获取服务实例（单例）"""
        if name not in self._instances:
            if name not in self._factories:
                raise KeyError(f"Service not registered: {name}")
            self._instances[name] = self._factories[name]()
        return self._instances[name]

    # ── Repository factories ──

    def _create_memory_repo(self) -> Any:
        from iris_memory.web.repositories.memory_repo import MemoryRepository
        return MemoryRepository(self._memory_service)

    def _create_kg_repo(self) -> Any:
        from iris_memory.web.repositories.kg_repo import KnowledgeGraphRepository
        return KnowledgeGraphRepository(self._memory_service)

    def _create_persona_repo(self) -> Any:
        from iris_memory.web.repositories.persona_repo import PersonaRepository
        return PersonaRepository(self._memory_service)

    def _create_emotion_repo(self) -> Any:
        from iris_memory.web.repositories.persona_repo import EmotionRepository
        return EmotionRepository(self._memory_service)

    def _create_session_repo(self) -> Any:
        from iris_memory.web.repositories.session_repo import SessionRepository
        return SessionRepository(self._memory_service)

    # ── Service factories ──

    def _create_dashboard_service(self) -> Any:
        from iris_memory.web.services.dashboard_service import DashboardService
        return DashboardService(
            self._memory_service,
            self.get("memory_repo"),
            self.get("session_repo"),
        )

    def _create_memory_web_service(self) -> Any:
        from iris_memory.web.services.memory_service import MemoryWebService
        return MemoryWebService(
            self._memory_service,
            self.get("memory_repo"),
        )

    def _create_kg_service(self) -> Any:
        from iris_memory.web.services.kg_service import KgWebService
        return KgWebService(self._memory_service)

    def _create_persona_service(self) -> Any:
        from iris_memory.web.services.persona_service import PersonaWebService
        return PersonaWebService(
            self.get("persona_repo"),
            self.get("emotion_repo"),
        )

    def _create_proactive_service(self) -> Any:
        from iris_memory.web.services.proactive_service import ProactiveWebService
        return ProactiveWebService(self._memory_service)

    def _create_io_service(self) -> Any:
        from iris_memory.web.services.io_service import IoService
        return IoService(self._memory_service)

    def _create_cooldown_service(self) -> Any:
        from iris_memory.web.services.cooldown_service import CooldownWebService
        return CooldownWebService(self._memory_service)

    def _create_llm_service(self) -> Any:
        from iris_memory.web.services.llm_service import LlmWebService
        return LlmWebService()

    def _create_config_service(self) -> Any:
        from iris_memory.web.services.config_service import ConfigWebService
        return ConfigWebService()

    def _create_system_service(self) -> Any:
        from iris_memory.web.services.system_service import SystemWebService
        return SystemWebService(self._memory_service)
