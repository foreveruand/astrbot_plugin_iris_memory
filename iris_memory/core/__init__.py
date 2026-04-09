"""Core module for iris memory"""

from iris_memory.config import (
    SCHEMA,
    AccessLevel,
    ConfigStore,
    config_events,
    get_store,
    init_store,
    reset_store,
)
from iris_memory.core.activity_config import (
    ACTIVITY_PRESETS,
    ActivityAwareConfigProvider,
    ActivityBasedPresets,
    GroupActivityLevel,
    GroupActivityTracker,
)
from iris_memory.core.provider_utils import (
    extract_provider_id,
    get_default_provider,
    get_provider_by_id,
    normalize_provider_id,
)
from iris_memory.core.service_container import ServiceContainer
from iris_memory.core.upgrade_evaluator import UpgradeEvaluator, UpgradeMode


def __getattr__(name: str):
    if name in ("BaseDetectionResult", "DetectionMode", "LLMEnhancedDetector"):
        from iris_memory.core.detection import (
            BaseDetectionResult,
            DetectionMode,
            LLMEnhancedDetector,
        )

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GroupActivityLevel",
    "ActivityBasedPresets",
    "ACTIVITY_PRESETS",
    "GroupActivityTracker",
    "ActivityAwareConfigProvider",
    "ServiceContainer",
    "UpgradeEvaluator",
    "UpgradeMode",
    "normalize_provider_id",
    "extract_provider_id",
    "get_provider_by_id",
    "get_default_provider",
    "BaseDetectionResult",
    "DetectionMode",
    "LLMEnhancedDetector",
    "ConfigStore",
    "get_store",
    "init_store",
    "reset_store",
    "SCHEMA",
    "AccessLevel",
    "config_events",
]
