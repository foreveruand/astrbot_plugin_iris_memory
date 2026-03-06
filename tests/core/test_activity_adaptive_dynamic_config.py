"""群聊自适应动态配置测试"""

from unittest.mock import Mock

from iris_memory.core.activity_config import (
    GroupActivityTracker,
    ActivityAwareConfigProvider,
    ACTIVITY_PRESETS,
    GroupActivityLevel,
)
from iris_memory.config import get_store, init_store, reset_store


def _build_user_config_with_advanced(**overrides):
    config = Mock()
    config.advanced = overrides
    return config


class TestActivityAdaptiveDynamicConfig:
    """覆盖群聊自适应下所有动态行为配置键"""

    def test_group_uses_activity_presets_when_enabled(self, monkeypatch):
        """启用自适应时，群聊应使用活跃度预设值"""
        reset_store()
        store = init_store(user_config=Mock())
        tracker = GroupActivityTracker()
        provider = ActivityAwareConfigProvider(tracker=tracker, enabled=True)

        monkeypatch.setattr(
            tracker,
            "get_activity_level",
            lambda group_id: GroupActivityLevel.ACTIVE,
        )

        group_id = "group-active"
        assert provider.get_cooldown_seconds(group_id) == ACTIVITY_PRESETS.cooldown_seconds["active"]
        assert provider.get_max_daily_replies(group_id) == ACTIVITY_PRESETS.max_daily_replies["active"]
        assert provider.get_batch_threshold_count(group_id) == ACTIVITY_PRESETS.batch_threshold_count["active"]
        assert provider.get_batch_threshold_interval(group_id) == ACTIVITY_PRESETS.batch_threshold_interval["active"]
        assert provider.get_daily_analysis_budget(group_id) == ACTIVITY_PRESETS.daily_analysis_budget["active"]
        assert provider.get_chat_context_count(group_id) == ACTIVITY_PRESETS.chat_context_count["active"]
        assert provider.get_reply_temperature(group_id) == ACTIVITY_PRESETS.reply_temperature["active"]

    def test_private_chat_uses_defaults_when_enabled(self):
        """私聊（group_id=None）应使用默认值"""
        reset_store()
        store = init_store(user_config=Mock())
        tracker = GroupActivityTracker()
        provider = ActivityAwareConfigProvider(tracker=tracker, enabled=True)

        assert provider.get_cooldown_seconds(None) == get_store().get("proactive_reply.cooldown_seconds", 60)
        assert provider.get_max_daily_replies(None) == get_store().get("proactive_reply.max_daily_replies", 20)
        assert provider.get_batch_threshold_count(None) == get_store().get("message_processing.batch_threshold_count", 20)
        assert provider.get_batch_threshold_interval(None) == get_store().get("message_processing.batch_threshold_interval", 300)
        assert provider.get_daily_analysis_budget(None) == get_store().get("image_analysis.daily_analysis_budget", 100)
        assert provider.get_chat_context_count(None) == get_store().get("llm_integration.chat_context_count", 15)
        assert provider.get_reply_temperature(None) == get_store().get("proactive_reply.reply_temperature", 0.7)

    def test_group_uses_defaults_when_adaptive_disabled(self):
        """禁用自适应时，群聊也应使用默认值"""
        reset_store()
        store = init_store(user_config=Mock())
        tracker = GroupActivityTracker()
        provider = ActivityAwareConfigProvider(tracker=tracker, enabled=False)

        group_id = "group-disabled"
        assert provider.get_cooldown_seconds(group_id) == get_store().get("proactive_reply.cooldown_seconds", 60)
        assert provider.get_max_daily_replies(group_id) == get_store().get("proactive_reply.max_daily_replies", 20)
        assert provider.get_batch_threshold_count(group_id) == get_store().get("message_processing.batch_threshold_count", 20)
        assert provider.get_batch_threshold_interval(group_id) == get_store().get("message_processing.batch_threshold_interval", 300)
        assert provider.get_daily_analysis_budget(group_id) == get_store().get("image_analysis.daily_analysis_budget", 100)
        assert provider.get_chat_context_count(group_id) == get_store().get("llm_integration.chat_context_count", 15)
        assert provider.get_reply_temperature(group_id) == get_store().get("proactive_reply.reply_temperature", 0.7)
