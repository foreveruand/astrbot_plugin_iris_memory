"""Persona module - 用户画像提取与协调"""

from iris_memory.persona.keyword_maps import ExtractionResult, KeywordMaps
from iris_memory.persona.rule_extractor import RuleExtractor
from iris_memory.persona.persona_coordinator import (
    PersonaCoordinator,
    PersonaConflictDetector,
    CoordinationStrategy,
    ConflictType,
)
from iris_memory.persona.persona_logger import PersonaLogger, persona_log
from iris_memory.persona.persona_batch_processor import (
    PersonaBatchProcessor,
    PersonaQueuedMessage,
    PersonaBatchStats,
)


def __getattr__(name: str):
    """Lazy import for modules that trigger heavy dependency chains."""
    if name == "PersonaExtractor":
        from iris_memory.persona.persona_extractor import PersonaExtractor
        return PersonaExtractor
    if name == "LLMExtractor":
        from iris_memory.persona.llm_extractor import LLMExtractor
        return LLMExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'ExtractionResult',
    'KeywordMaps',
    'PersonaExtractor',
    'RuleExtractor',
    'LLMExtractor',
    'PersonaCoordinator',
    'PersonaConflictDetector',
    'CoordinationStrategy',
    'ConflictType',
    'PersonaLogger',
    'persona_log',
    'PersonaBatchProcessor',
    'PersonaQueuedMessage',
    'PersonaBatchStats',
]
