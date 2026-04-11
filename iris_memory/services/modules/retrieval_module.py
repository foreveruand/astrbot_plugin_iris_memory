"""
检索模块 — 封装 MemoryRetrievalEngine 的创建和配置

RetrievalEngine 内部已经聚合了 Reranker 和 RetrievalRouter。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from iris_memory.config.events import config_events
from iris_memory.utils.logger import get_logger

if TYPE_CHECKING:
    from iris_memory.retrieval.retrieval_engine import MemoryRetrievalEngine

logger = get_logger("module.retrieval")


class RetrievalModule:
    """检索模块"""

    def __init__(self) -> None:
        self._retrieval_engine: MemoryRetrievalEngine | None = None
        self._cfg: Any = None
        self._unsubscribe: Any = None

    @property
    def retrieval_engine(self) -> MemoryRetrievalEngine | None:
        return self._retrieval_engine

    def initialize(
        self,
        chroma_manager: Any,
        rif_scorer: Any,
        emotion_analyzer: Any,
        session_manager: Any,
        llm_retrieval_router: Any = None,
    ) -> None:
        """初始化检索引擎"""
        from iris_memory.retrieval.retrieval_engine import MemoryRetrievalEngine

        self._retrieval_engine = MemoryRetrievalEngine(
            chroma_manager=chroma_manager,
            rif_scorer=rif_scorer,
            emotion_analyzer=emotion_analyzer,
            session_manager=session_manager,
            llm_retrieval_router=llm_retrieval_router,
        )
        logger.debug("RetrievalModule initialized")

    def set_kg_module(self, kg_module: Any) -> None:
        """注入知识图谱模块到检索引擎"""
        if self._retrieval_engine:
            self._retrieval_engine.set_kg_module(kg_module)
            logger.debug("KG module injected into RetrievalEngine")

    def _subscribe_config_events(self, cfg: Any) -> None:
        """订阅配置变更事件"""
        # 取消之前的订阅
        if self._unsubscribe:
            self._unsubscribe()

        # 订阅 retrieval section 的配置变更
        self._unsubscribe = config_events.on_section(
            "retrieval", self._on_config_change
        )
        logger.debug("RetrievalModule subscribed to config changes")

    def _on_config_change(self, key: str, old_value: Any, new_value: Any) -> None:
        """配置变更回调"""
        logger.info(
            f"Retrieval config changed: {key} from {old_value} to {new_value}, "
            "applying new config"
        )
        if self._cfg and self._retrieval_engine:
            self.apply_config(self._cfg)

    def apply_config(self, cfg: Any) -> None:
        """应用配置到检索引擎"""
        from iris_memory.config import get_store

        # 保存 cfg 引用以便配置变更回调使用
        self._cfg = cfg

        # 订阅配置变更事件（首次调用时）
        if self._unsubscribe is None:
            self._subscribe_config_events(cfg)

        if self._retrieval_engine:
            self._retrieval_engine.set_config(
                {
                    "max_context_memories": cfg.get("memory.max_context_memories", 10),
                    "enable_time_aware": get_store().get(
                        "llm_integration.enable_time_aware"
                    ),
                    "enable_emotion_aware": get_store().get(
                        "llm_integration.enable_emotion_aware"
                    ),
                    "enable_token_budget": cfg.get("basic.enable_inject", True),
                    "token_budget": cfg.get("llm_integration.token_budget", 512),
                    "coordination_strategy": get_store().get(
                        "llm_integration.coordination_strategy"
                    ),
                    "include_group_private_in_private_query": get_store().get(
                        "retrieval.include_group_private_in_private_query", False
                    ),
                }
            )
