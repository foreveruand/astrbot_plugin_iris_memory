"""
主动回复限制检查

管理冷却时间、每日限制、连续回复限制、启动冷却等约束条件。
从 ProactiveReplyManager 中拆分，减少文件行数。
"""
import asyncio
import time
from datetime import date
from typing import Any, Dict, Optional, List, TYPE_CHECKING

from iris_memory.utils.logger import get_logger
from iris_memory.utils.command_utils import SessionKeyBuilder
from iris_memory.utils.bounded_dict import BoundedDict
from iris_memory.core.constants import UrgencyCooldownMultiplier

if TYPE_CHECKING:
    from iris_memory.core.config_manager import ConfigManager

logger = get_logger("proactive_constraints")


class ProactiveConstraints:
    """主动回复限制检查
    
    封装所有回复约束逻辑：冷却时间、每日限制、连续回复限制、启动冷却。
    """
    
    def __init__(
        self,
        config_manager: Optional['ConfigManager'] = None,
        default_cooldown: int = 60,
        default_max_daily: int = 20,
    ):
        self._config_manager = config_manager
        self._default_cooldown = default_cooldown
        self._default_max_daily = default_max_daily
        
        # 状态跟踪（有界，防止内存无限增长）
        self.last_reply_time: BoundedDict[str, float] = BoundedDict(max_size=2000)
        self.daily_reply_count: BoundedDict[str, int] = BoundedDict(max_size=2000)
        
        # 连续回复限制
        self._recent_replies: BoundedDict[str, List[float]] = BoundedDict(max_size=500)
        self.MAX_CONSECUTIVE_REPLIES: int = 3
        self.CONSECUTIVE_WINDOW: int = 300  # 5分钟窗口（秒）
        
        # 每日计数重置日期跟踪
        self._last_reset_date: date = date.today()
        
        # 启动冷却期
        self._startup_time: Optional[float] = None
        self.STARTUP_COOLDOWN_SECONDS: int = 120
    
    def get_cooldown_seconds(self, group_id: Optional[str] = None) -> int:
        """获取冷却时间"""
        if self._config_manager:
            return self._config_manager.get_cooldown_seconds(group_id)
        return self._default_cooldown
    
    def get_max_daily_replies(self, group_id: Optional[str] = None) -> int:
        """获取每日最大回复数"""
        if self._config_manager:
            return self._config_manager.get_max_daily_replies(group_id)
        return self._default_max_daily
    
    def is_in_cooldown(
        self,
        session_key: str,
        group_id: Optional[str] = None,
        urgency: Optional[str] = None,
    ) -> bool:
        """检查是否在冷却中
        
        支持基于 urgency 的动态冷却：
        - critical: 冷却时间 × 0.5（紧急事件缩短冷却）
        - high: 冷却时间 × 0.75
        - medium: 冷却时间 × 1.0（默认）
        - low: 冷却时间 × 1.5
        """
        if session_key not in self.last_reply_time:
            return False
        
        elapsed = asyncio.get_running_loop().time() - self.last_reply_time[session_key]
        base_cooldown = self.get_cooldown_seconds(group_id)
        
        urgency_multiplier = {
            "critical": UrgencyCooldownMultiplier.CRITICAL,
            "high": UrgencyCooldownMultiplier.HIGH,
            "medium": UrgencyCooldownMultiplier.MEDIUM,
            "low": UrgencyCooldownMultiplier.LOW,
        }
        multiplier = urgency_multiplier.get(urgency, 1.0)
        effective_cooldown = base_cooldown * multiplier
        
        return elapsed < effective_cooldown
    
    def check_daily_reset(self) -> None:
        """惰性检查是否跨日，跨日则重置每日计数"""
        today = date.today()
        if self._last_reset_date != today:
            self.daily_reply_count.clear()
            self._last_reset_date = today
            logger.debug("Daily proactive reply counts reset (new day)")
    
    def is_daily_limit_reached(self, user_id: str, group_id: Optional[str] = None) -> bool:
        """检查是否达到每日限制"""
        self.check_daily_reset()
        count_key = SessionKeyBuilder.build(user_id, group_id)
        count = self.daily_reply_count.get(count_key, 0)
        return count >= self.get_max_daily_replies(group_id)
    
    def is_consecutive_limit_reached(self, session_key: str) -> bool:
        """检查是否达到连续回复限制
        
        在 CONSECUTIVE_WINDOW 时间窗口内，同一会话最多允许
        MAX_CONSECUTIVE_REPLIES 次主动回复。
        """
        now = time.time()
        replies = self._recent_replies.get(session_key, [])
        replies = [t for t in replies if now - t < self.CONSECUTIVE_WINDOW]
        self._recent_replies[session_key] = replies
        return len(replies) >= self.MAX_CONSECUTIVE_REPLIES
    
    def record_reply_time(self, session_key: str) -> None:
        """记录一次主动回复时间"""
        now = time.time()
        replies = self._recent_replies.get(session_key, [])
        replies = [t for t in replies if now - t < self.CONSECUTIVE_WINDOW]
        replies.append(now)
        self._recent_replies[session_key] = replies
    
    def is_in_startup_cooldown(self) -> bool:
        """检查是否在启动冷却期内"""
        if self._startup_time is None:
            return False
        elapsed = time.time() - self._startup_time
        return elapsed < self.STARTUP_COOLDOWN_SECONDS
    
    def increment_daily_count(self, user_id: str, group_id: Optional[str] = None) -> None:
        """增加每日回复计数"""
        count_key = SessionKeyBuilder.build(user_id, group_id)
        self.daily_reply_count[count_key] = self.daily_reply_count.get(count_key, 0) + 1
    
    def reset_daily_counts(self) -> None:
        """重置每日计数"""
        self.daily_reply_count.clear()
        logger.debug("Daily proactive reply counts reset")
