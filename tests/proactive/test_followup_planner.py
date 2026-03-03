"""test_followup_planner.py - FollowUp 跟进规划器测试"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from iris_memory.proactive.config import FollowUpConfig, ProactiveConfig
from iris_memory.proactive.followup_planner import (
    FollowUpPlanner,
    build_followup_prompt,
    parse_followup_response,
)
from iris_memory.proactive.models import (
    FollowUpDecision,
    FollowUpExpectation,
    FollowUpReplyType,
    ProactiveReplyResult,
)
from iris_memory.proactive.storage.expectation_store import ExpectationStore


@pytest.fixture
def config() -> ProactiveConfig:
    return ProactiveConfig(
        enabled=True,
        followup_enabled=True,
        followup_window_seconds=120,
        max_followup_count=2,
        followup=FollowUpConfig(
            short_window_seconds=2,
            fallback_to_rule_on_llm_error=True,
        ),
    )


@pytest.fixture
def store() -> ExpectationStore:
    return ExpectationStore()


@pytest.fixture
def reply_callback() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def llm_callback() -> AsyncMock:
    return AsyncMock(return_value=FollowUpDecision(
        should_reply=True,
        reason="用户有回应",
        reply_type=FollowUpReplyType.CONTINUE_TOPIC,
        suggested_direction="继续聊这个",
    ))


@pytest.fixture
def planner(
    config: ProactiveConfig,
    store: ExpectationStore,
    reply_callback: AsyncMock,
    llm_callback: AsyncMock,
) -> FollowUpPlanner:
    return FollowUpPlanner(
        config=config,
        expectation_store=store,
        on_followup_reply=reply_callback,
        on_llm_decide=llm_callback,
    )


class TestCreateExpectation:
    """创建跟进期待测试"""

    @pytest.mark.asyncio
    async def test_create_basic(self, planner: FollowUpPlanner) -> None:
        exp = planner.create_expectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="你好呀",
            bot_reply_summary="我回复了你好",
        )
        assert exp is not None
        assert exp.group_id == "g1"
        assert exp.trigger_user_id == "u1"
        assert exp.followup_count == 0
        assert planner.has_active_expectation("g1")

    def test_disabled_returns_none(
        self, config: ProactiveConfig, store: ExpectationStore
    ) -> None:
        config.followup_enabled = False
        planner = FollowUpPlanner(config=config, expectation_store=store)
        exp = planner.create_expectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="hello",
        )
        assert exp is None

    def test_max_count_returns_none(self, planner: FollowUpPlanner) -> None:
        exp = planner.create_expectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="hello",
            followup_count=2,  # = max_followup_count
        )
        assert exp is None

    @pytest.mark.asyncio
    async def test_replaces_old_expectation(self, planner: FollowUpPlanner) -> None:
        exp1 = planner.create_expectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="msg1",
            bot_reply_summary="reply1",
        )
        exp2 = planner.create_expectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="msg2",
            bot_reply_summary="reply2",
        )
        assert exp1.expectation_id != exp2.expectation_id
        assert planner.active_expectation_count == 1


class TestOnUserMessage:
    """用户消息处理测试"""

    @pytest.mark.asyncio
    async def test_aggregates_trigger_user_message(self, planner: FollowUpPlanner) -> None:
        planner.create_expectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="hello",
        )
        result = planner.on_user_message(
            user_id="u1",
            group_id="g1",
            message="thanks!",
            sender_name="User1",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_ignores_other_user(self, planner: FollowUpPlanner) -> None:
        planner.create_expectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="hello",
        )
        result = planner.on_user_message(
            user_id="u2",
            group_id="g1",
            message="hey",
        )
        assert result is False

    def test_no_expectation_returns_false(self, planner: FollowUpPlanner) -> None:
        result = planner.on_user_message(user_id="u1", group_id="g1", message="x")
        assert result is False

    def test_expired_window_returns_false(
        self,
        config: ProactiveConfig,
        store: ExpectationStore,
    ) -> None:
        planner = FollowUpPlanner(config=config, expectation_store=store)
        # Manually create expired expectation
        exp = FollowUpExpectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="hello",
            followup_window_end=datetime.now() - timedelta(seconds=1),
        )
        store.put(exp)

        result = planner.on_user_message(user_id="u1", group_id="g1", message="x")
        assert result is False
        assert not planner.has_active_expectation("g1")


class TestClearExpectation:
    """清除期待测试"""

    @pytest.mark.asyncio
    async def test_clear_existing(self, planner: FollowUpPlanner) -> None:
        planner.create_expectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="hello",
        )
        planner.clear_expectation("g1")
        assert not planner.has_active_expectation("g1")

    def test_clear_nonexistent(self, planner: FollowUpPlanner) -> None:
        # Should not raise
        planner.clear_expectation("nonexistent")


class TestRuleFallbackDecision:
    """规则降级判断测试"""

    def test_no_messages_no_reply(self, planner: FollowUpPlanner) -> None:
        exp = FollowUpExpectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="hello",
            followup_window_end=datetime.now() + timedelta(minutes=2),
        )
        decision = planner._rule_fallback_decision(exp)
        assert decision is not None
        assert decision.should_reply is False

    def test_short_reply_no_followup(self, planner: FollowUpPlanner) -> None:
        exp = FollowUpExpectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="hello",
            followup_window_end=datetime.now() + timedelta(minutes=2),
            aggregated_messages=[{"content": "嗯"}],
        )
        decision = planner._rule_fallback_decision(exp)
        assert decision.should_reply is False

    def test_substantive_reply_triggers_followup(self, planner: FollowUpPlanner) -> None:
        exp = FollowUpExpectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="hello",
            followup_window_end=datetime.now() + timedelta(minutes=2),
            aggregated_messages=[{"content": "我觉得这个问题很有趣"}],
        )
        decision = planner._rule_fallback_decision(exp)
        assert decision.should_reply is True
        assert decision.reply_type == FollowUpReplyType.CONTINUE_TOPIC


class TestLLMDecision:
    """LLM 判断测试"""

    @pytest.mark.asyncio
    async def test_llm_callback_invoked(
        self,
        planner: FollowUpPlanner,
        llm_callback: AsyncMock,
    ) -> None:
        exp = FollowUpExpectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="hello",
            followup_window_end=datetime.now() + timedelta(minutes=2),
            aggregated_messages=[{"content": "谢谢"}],
        )
        decision = await planner._get_llm_decision(exp)
        llm_callback.assert_called_once()
        assert decision is not None
        assert decision.should_reply is True

    @pytest.mark.asyncio
    async def test_llm_error_fallback_to_rule(
        self,
        config: ProactiveConfig,
        store: ExpectationStore,
    ) -> None:
        """LLM 错误时降级到规则判断"""
        error_callback = AsyncMock(side_effect=RuntimeError("LLM down"))
        planner = FollowUpPlanner(
            config=config,
            expectation_store=store,
            on_llm_decide=error_callback,
        )
        exp = FollowUpExpectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="hello",
            followup_window_end=datetime.now() + timedelta(minutes=2),
            aggregated_messages=[{"content": "我觉得很有道理，继续说说"}],
        )
        decision = await planner._get_llm_decision(exp)
        assert decision is not None
        assert decision.should_reply is True  # 规则降级

    @pytest.mark.asyncio
    async def test_llm_error_no_fallback(
        self,
        store: ExpectationStore,
    ) -> None:
        """fallback_to_rule_on_llm_error=False 时 LLM 错误返回 None"""
        config = ProactiveConfig(
            followup_enabled=True,
            followup=FollowUpConfig(fallback_to_rule_on_llm_error=False),
        )
        error_callback = AsyncMock(side_effect=RuntimeError("LLM down"))
        planner = FollowUpPlanner(
            config=config,
            expectation_store=store,
            on_llm_decide=error_callback,
        )
        exp = FollowUpExpectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="hello",
            followup_window_end=datetime.now() + timedelta(minutes=2),
            aggregated_messages=[{"content": "okay"}],
        )
        decision = await planner._get_llm_decision(exp)
        assert decision is None

    @pytest.mark.asyncio
    async def test_no_llm_callback_rule_fallback(
        self,
        config: ProactiveConfig,
        store: ExpectationStore,
    ) -> None:
        """没有 LLM 回调时使用规则降级"""
        planner = FollowUpPlanner(
            config=config,
            expectation_store=store,
            on_llm_decide=None,
        )
        exp = FollowUpExpectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="hello",
            followup_window_end=datetime.now() + timedelta(minutes=2),
            aggregated_messages=[{"content": "这很有意思，我想了解更多"}],
        )
        decision = await planner._get_llm_decision(exp)
        assert decision is not None
        assert decision.should_reply is True


class TestBuildFollowupPrompt:
    """构建 FollowUp Prompt 测试"""

    def test_basic_prompt(self) -> None:
        exp = FollowUpExpectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="你好呀",
            followup_window_end=datetime.now() + timedelta(minutes=2),
            aggregated_messages=[
                {"sender_name": "张三", "content": "谢谢你的回复"},
            ],
            recent_context=[
                {"sender_name": "张三", "content": "大家好"},
            ],
        )
        prompt = build_followup_prompt(exp)
        assert "你好呀" in prompt
        assert "谢谢你的回复" in prompt
        assert "大家好" in prompt

    def test_empty_messages(self) -> None:
        exp = FollowUpExpectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="reply",
            followup_window_end=datetime.now() + timedelta(minutes=2),
        )
        prompt = build_followup_prompt(exp)
        assert "无用户发言" in prompt
        assert "无上下文" in prompt


class TestParseFollowupResponse:
    """解析 FollowUp LLM 响应测试"""

    def test_valid_json(self) -> None:
        response = '{"should_reply": true, "reason": "用户回应了", "reply_type": "continue_topic", "suggested_direction": "继续聊"}'
        decision = parse_followup_response(response)
        assert decision is not None
        assert decision.should_reply is True
        assert decision.reason == "用户回应了"
        assert decision.reply_type == FollowUpReplyType.CONTINUE_TOPIC
        assert decision.suggested_direction == "继续聊"

    def test_json_in_markdown(self) -> None:
        response = '好的，这是我的判断：\n```json\n{"should_reply": false, "reason": "用户在和别人聊天"}\n```'
        decision = parse_followup_response(response)
        # parse_followup_response extracts first {...} instead of markdown block
        assert decision is not None
        assert decision.should_reply is False

    def test_json_with_surrounding_text(self) -> None:
        response = '根据分析:{"should_reply": true, "reason": "test", "reply_type": "question"}结束'
        decision = parse_followup_response(response)
        assert decision is not None
        assert decision.should_reply is True
        assert decision.reply_type == FollowUpReplyType.QUESTION

    def test_invalid_json(self) -> None:
        decision = parse_followup_response("this is not json")
        assert decision is None

    def test_empty_response(self) -> None:
        assert parse_followup_response("") is None
        assert parse_followup_response(None) is None

    def test_unknown_reply_type_defaults(self) -> None:
        response = '{"should_reply": true, "reason": "ok", "reply_type": "unknown_type"}'
        decision = parse_followup_response(response)
        assert decision is not None
        assert decision.reply_type == FollowUpReplyType.ACKNOWLEDGE

    def test_missing_optional_fields(self) -> None:
        response = '{"should_reply": false}'
        decision = parse_followup_response(response)
        assert decision is not None
        assert decision.should_reply is False
        assert decision.reason == ""
        assert decision.suggested_direction == ""


class TestFollowUpPlannerClose:
    """关闭测试"""

    @pytest.mark.asyncio
    async def test_close(self, planner: FollowUpPlanner) -> None:
        planner.create_expectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="hello",
        )
        await planner.close()
        assert planner.active_expectation_count == 0


class TestBuildFollowupReply:
    """构建跟进回复结果测试"""

    def test_build_reply(self) -> None:
        exp = FollowUpExpectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="你好",
            followup_window_end=datetime.now() + timedelta(minutes=2),
            aggregated_messages=[
                {"sender_name": "用户", "content": "谢谢呀"},
            ],
        )
        decision = FollowUpDecision(
            should_reply=True,
            reason="用户回应了",
            reply_type=FollowUpReplyType.ACKNOWLEDGE,
            suggested_direction="回复感谢",
        )
        result = FollowUpPlanner._build_followup_reply(exp, decision)
        assert isinstance(result, ProactiveReplyResult)
        assert result.source == "followup"
        assert "跟进回复" in result.reason
        assert "你好" in result.trigger_prompt
        assert result.target_user == "u1"
