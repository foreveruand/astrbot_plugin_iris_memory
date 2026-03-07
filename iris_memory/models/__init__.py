"""Models module for iris memory

核心数据模型层，定义系统的实体结构。
"""

from iris_memory.models.memory import Memory
from iris_memory.models.emotion_state import EmotionalState, CurrentEmotionState, EmotionalTrajectory, TrendType
from iris_memory.models.user_persona import UserPersona
from iris_memory.models.persona_change import PersonaChangeRecord
from iris_memory.models.persona_view import build_injection_view
from iris_memory.models.persona_extraction_applier import (
    apply_extraction_result as apply_persona_extraction,
)
from iris_memory.models.protection import ProtectionFlag, ProtectionMixin, ProtectionRules

__all__ = [
    "Memory",
    "EmotionalState",
    "CurrentEmotionState",
    "EmotionalTrajectory",
    "TrendType",
    "UserPersona",
    "PersonaChangeRecord",
    "build_injection_view",
    "apply_persona_extraction",
    "ProtectionFlag",
    "ProtectionMixin",
    "ProtectionRules",
]
