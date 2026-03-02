"""
冷却模块 — Feature Module 封装

将 CooldownManager 封装为 Feature Module，
遵循项目 Facade + 组合模式架构。
"""
from __future__ import annotations

from typing import Optional

from iris_memory.utils.logger import get_logger

logger = get_logger("module.cooldown")


class CooldownModule:
    """冷却模块

    封装群冷却管理器，提供统一的模块化接口。
    无需异步初始化，CooldownManager 在构造时即可用。
    """

    def __init__(self) -> None:
        from iris_memory.cooldown.cooldown_manager import CooldownManager
        self._cooldown_manager: CooldownManager = CooldownManager()
        logger.debug("CooldownModule initialized")

    @property
    def cooldown_manager(self) -> "CooldownManager":
        """获取冷却管理器"""
        return self._cooldown_manager

    def is_active(self, group_id: Optional[str]) -> bool:
        """检查群是否处于冷却中

        便捷代理方法，供 ProactiveModule 等外部模块直接调用。

        Args:
            group_id: 群聊 ID，None 时返回 False（私聊不受冷却影响）

        Returns:
            bool: 是否处于冷却中
        """
        if not group_id:
            return False
        return self._cooldown_manager.is_active(group_id)
