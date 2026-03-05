"""
主动回复 v3 配置 - 简化版

所有配置统一从全局 defaults.py 读取，减少调用链条复杂性。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iris_memory.core.defaults import ProactiveReplyDefaults


class ProactiveConfig:
    """主动回复配置 - 直接从全局 defaults 读取
    
    不再使用复杂的 dataclass 嵌套，所有属性直接映射到 defaults.py
    """
    
    def __init__(self, defaults: "ProactiveReplyDefaults") -> None:
        """从全局 defaults 初始化配置
        
        Args:
            defaults: ProactiveReplyDefaults 实例
        """
        # 总开关
        self.enabled: bool = getattr(defaults, 'enable', False)
        self.signal_queue_enabled: bool = True
        self.followup_enabled: bool = True
        
        # 核心功能开关
        self.followup_after_all_replies: bool = defaults.followup_after_all_replies
        self.group_whitelist_mode: bool = defaults.group_whitelist_mode
        self.proactive_mode: str = getattr(defaults, 'proactive_mode', 'rule')
        
        # 时间窗口配置
        self.followup_window_seconds: int = defaults.followup_window_seconds
        self.cooldown_seconds: int = defaults.cooldown_seconds
        
        # 限制配置
        self.max_followup_count: int = defaults.max_followup_count
        self.max_daily_replies: int = defaults.max_daily_replies
        self.max_daily_per_user: int = defaults.max_daily_per_user
        self.max_reply_tokens: int = defaults.max_reply_tokens
        
        # 回复参数
        self.reply_temperature: float = defaults.reply_temperature
        
        # 白名单
        self.group_whitelist: list = list(defaults.group_whitelist)
        
        # 静音时段
        self.quiet_hours: list = list(defaults.quiet_hours)
        self.quiet_hours_activity_exempt_minutes: int = getattr(
            defaults, 'quiet_hours_activity_exempt_minutes', 20
        )
        self.timezone_offset: int = getattr(defaults, 'timezone_offset', 8)
        
        # SignalQueue 高级配置 (直接扁平化，不再嵌套)
        self.signal_check_interval_seconds: int = defaults.signal_check_interval_seconds
        self.signal_silence_timeout_seconds: int = defaults.signal_silence_timeout_seconds
        self.signal_min_silence_seconds: int = defaults.signal_min_silence_seconds
        self.signal_ttl_emotion_high: int = defaults.signal_ttl_emotion_high
        self.signal_ttl_rule_match: int = defaults.signal_ttl_rule_match
        self.signal_weight_direct_reply: float = defaults.signal_weight_direct_reply
        self.signal_weight_llm_confirm: float = defaults.signal_weight_llm_confirm
        self.signal_max_signals_per_group: int = getattr(
            defaults, 'signal_max_signals_per_group', 50
        )
        
        # FollowUp 高级配置 (直接扁平化)
        self.followup_short_window_seconds: int = defaults.followup_short_window_seconds
        self.followup_llm_max_tokens: int = defaults.followup_llm_max_tokens
        self.followup_llm_temperature: float = defaults.followup_llm_temperature
        self.followup_fallback_to_rule_on_llm_error: bool = defaults.followup_fallback_to_rule
    
    def to_dict(self) -> dict:
        """导出配置为字典（用于调试）"""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
    
    def __repr__(self) -> str:
        return f"ProactiveConfig(enabled={self.enabled}, followup={self.followup_enabled})"


# 保持向后兼容的导出
SignalQueueConfig = None  # 已废弃，配置直接扁平化到 ProactiveConfig
FollowUpConfig = None     # 已废弃，配置直接扁平化到 ProactiveConfig
