"""
群冷却模块 — 暂停群聊主动回复

提供群聊冷却管理能力，在冷却期内暂停 AI 的主动回复。
支持用户手动触发和 LLM 工具调用两种方式。
"""

from iris_memory.cooldown.cooldown_manager import CooldownManager
from iris_memory.cooldown.cooldown_state import CooldownState

__all__ = ["CooldownState", "CooldownManager"]
