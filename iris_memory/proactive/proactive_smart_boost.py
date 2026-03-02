"""
主动回复智能增强管理

管理智能增强窗口、乘数计算和决策调整。
从 ProactiveReplyManager 中拆分，减少文件行数。
"""
import time
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

from iris_memory.utils.logger import get_logger
from iris_memory.utils.command_utils import SessionKeyBuilder
from iris_memory.utils.bounded_dict import BoundedDict
from iris_memory.proactive.proactive_reply_detector import (
    ProactiveReplyDecision, ReplyUrgency,
)

if TYPE_CHECKING:
    from iris_memory.core.config_manager import ConfigManager
    from iris_memory.proactive.llm_proactive_reply_detector import LLMReplyDecision

logger = get_logger("proactive_smart_boost")


class SmartBoostManager:
    """主动回复智能增强管理
    
    在用户活跃期间提升回复概率或缩短回复延迟。
    窗口基于用户发言时间而非 Bot 回复时间，
    避免 Bot 自身回复不断刷新窗口导致"滚雪球"式连续回复。
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional['ConfigManager'] = None,
        stats: Optional[Dict[str, int]] = None,
    ):
        self._config = config or {}
        self._config_manager = config_manager
        self._last_user_message_time: BoundedDict[str, float] = BoundedDict(max_size=2000)
        
        # 共享统计（由外部传入，或自行创建）
        self.stats = stats if stats is not None else {
            "smart_boost_activations": 0,
            "smart_boost_delay_reductions": 0,
        }
    
    @property
    def enabled(self) -> bool:
        """获取智能增强开关"""
        if self._config_manager:
            return self._config_manager.smart_boost_enabled
        return self._config.get("smart_boost_enabled", False)
    
    @property
    def window_seconds(self) -> int:
        """获取智能增强窗口时间（秒）"""
        if self._config_manager:
            return self._config_manager.smart_boost_window_seconds
        return self._config.get("smart_boost_window_seconds", 60)
    
    @property
    def multiplier(self) -> float:
        """获取智能增强分数乘数"""
        if self._config_manager:
            return self._config_manager.smart_boost_score_multiplier
        return self._config.get("smart_boost_score_multiplier", 1.2)
    
    @property
    def threshold(self) -> float:
        """获取智能增强回复阈值"""
        if self._config_manager:
            return self._config_manager.smart_boost_reply_threshold
        return self._config.get("smart_boost_reply_threshold", 0.5)
    
    def record_user_message(self, user_id: str, group_id: Optional[str] = None) -> None:
        """记录用户发言时间"""
        key = SessionKeyBuilder.build(user_id, group_id)
        self._last_user_message_time[key] = time.time()
    
    def is_in_boost_window(self, user_id: str, group_id: Optional[str] = None) -> bool:
        """检查是否在智能增强窗口内"""
        if not self.enabled:
            return False
        if self._config_manager:
            mode = self._config_manager.proactive_mode
            if mode not in ("llm", "hybrid"):
                return False
        key = SessionKeyBuilder.build(user_id, group_id)
        last_time = self._last_user_message_time.get(key)
        if last_time is None:
            return False
        elapsed = time.time() - last_time
        return elapsed < self.window_seconds
    
    def get_boost_multiplier(self, user_id: str, group_id: Optional[str] = None) -> float:
        """获取当前智能增强乘数（线性衰减）
        
        Returns:
            乘数值，不在增强窗口内返回 1.0
        """
        if not self.is_in_boost_window(user_id, group_id):
            return 1.0
        key = SessionKeyBuilder.build(user_id, group_id)
        last_time = self._last_user_message_time.get(key)
        if last_time is None:
            return 1.0
        elapsed = time.time() - last_time
        decay = 1 - (elapsed / self.window_seconds)
        return 1.0 + (self.multiplier - 1.0) * decay
    
    def apply(
        self,
        decision: Union[ProactiveReplyDecision, "LLMReplyDecision"],
        user_id: str,
        group_id: Optional[str] = None,
    ) -> Union[ProactiveReplyDecision, "LLMReplyDecision"]:
        """对检测决策应用智能增强
        
        - 若决策已为 should_reply=True：缩短建议延迟
        - 若决策为不回复但分数可被增强到阈值以上：翻转为回复
        
        Args:
            decision: 检测器返回的决策
            user_id: 用户 ID
            group_id: 群聊 ID
            
        Returns:
            可能被修改的决策对象（原地修改）
        """
        boost_multiplier = self.get_boost_multiplier(user_id, group_id)
        if boost_multiplier <= 1.0:
            return decision
        
        # 已经要回复 → 缩短延迟
        if decision.should_reply:
            return self._reduce_delay(decision, boost_multiplier)
        
        # 不回复 → 尝试增强分数
        return self._try_boost_score(decision, boost_multiplier)
    
    def _reduce_delay(
        self,
        decision: Union[ProactiveReplyDecision, "LLMReplyDecision"],
        boost_multiplier: float,
    ) -> Union[ProactiveReplyDecision, "LLMReplyDecision"]:
        """缩短已确认回复的延迟"""
        original_delay = decision.suggested_delay
        decision.suggested_delay = max(0, int(original_delay / boost_multiplier))
        if original_delay != decision.suggested_delay:
            self.stats["smart_boost_delay_reductions"] += 1
            logger.debug(
                f"Smart boost reduced delay: {original_delay}s → "
                f"{decision.suggested_delay}s (×{boost_multiplier:.2f})"
            )
        return decision
    
    def _try_boost_score(
        self,
        decision: Union[ProactiveReplyDecision, "LLMReplyDecision"],
        boost_multiplier: float,
    ) -> Union[ProactiveReplyDecision, "LLMReplyDecision"]:
        """尝试增强回复分数并翻转决策"""
        reply_score = 0.0
        reply_context = getattr(decision, 'reply_context', None)
        if isinstance(reply_context, dict):
            reply_score = reply_context.get("reply_score", 0.0)
        
        if reply_score <= 0 and hasattr(decision, 'confidence'):
            reply_score = getattr(decision, 'confidence', 0.0) * 0.5
        
        if reply_score <= 0:
            return decision
        
        boosted_score = reply_score * boost_multiplier
        
        if boosted_score >= self.threshold:
            decision.should_reply = True
            if boosted_score >= 0.5:
                decision.urgency = ReplyUrgency.HIGH
                decision.suggested_delay = max(5, int(10 / boost_multiplier))
            elif boosted_score >= 0.4:
                decision.urgency = ReplyUrgency.MEDIUM
                decision.suggested_delay = max(10, int(20 / boost_multiplier))
            else:
                decision.urgency = ReplyUrgency.MEDIUM
                decision.suggested_delay = max(15, int(30 / boost_multiplier))
            
            original_reason = decision.reason or ""
            decision.reason = (
                f"{original_reason} + smart_boost"
                f"(×{boost_multiplier:.2f}, {reply_score:.2f}→{boosted_score:.2f})"
            )
            
            if isinstance(reply_context, dict):
                reply_context["reply_score"] = boosted_score
                reply_context["smart_boost_applied"] = True
                reply_context["smart_boost_multiplier"] = boost_multiplier
            
            logger.debug(
                f"Smart boost activated: score {reply_score:.2f} → "
                f"{boosted_score:.2f} (×{boost_multiplier:.2f})"
            )
            self.stats["smart_boost_activations"] += 1
        
        return decision
