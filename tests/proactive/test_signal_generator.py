"""test_signal_generator.py - 信号生成器测试"""

from __future__ import annotations

import pytest

from iris_memory.proactive.config import ProactiveConfig
from iris_memory.proactive.models import Signal, SignalType
from iris_memory.proactive.signal_generator import (
    ATTENTION_KEYWORDS,
    EMOTION_NEGATIVE,
    EMOTION_POSITIVE,
    MENTION_PATTERNS,
    QUESTION_KEYWORDS,
    SHORT_CONFIRM_PATTERNS,
    SignalGenerator,
)


@pytest.fixture
def config() -> ProactiveConfig:
    return ProactiveConfig()


@pytest.fixture
def generator(config: ProactiveConfig) -> SignalGenerator:
    return SignalGenerator(config)


def _gen(gen: SignalGenerator, text: str, emotion: float = 0.0):
    return gen.generate(
        text=text,
        user_id="u1",
        group_id="g1",
        session_key="u1:g1",
        emotion_intensity=emotion,
    )


class TestNegativeFilter:
    """负向检测 - 应被过滤的消息"""

    def test_empty_string(self, generator: SignalGenerator) -> None:
        assert _gen(generator, "") == []

    def test_whitespace_only(self, generator: SignalGenerator) -> None:
        assert _gen(generator, "   ") == []

    @pytest.mark.parametrize("text", SHORT_CONFIRM_PATTERNS[:6])
    def test_short_confirm_filtered(self, generator: SignalGenerator, text: str) -> None:
        assert _gen(generator, text) == []

    def test_short_confirm_with_trailing_punctuation(self, generator: SignalGenerator) -> None:
        assert _gen(generator, "好的。") == []
        assert _gen(generator, "行!") == []

    def test_emoji_only_filtered(self, generator: SignalGenerator) -> None:
        assert _gen(generator, "😊") == []
        assert _gen(generator, "😊😊😊") == []

    def test_bracket_emoji_filtered(self, generator: SignalGenerator) -> None:
        assert _gen(generator, "[微笑]") == []
        assert _gen(generator, "[微笑][开心]") == []


class TestQuestionDetection:
    """疑问词检测"""

    def test_question_keyword(self, generator: SignalGenerator) -> None:
        signals = _gen(generator, "这个是什么呀？")
        assert any(s.signal_type == SignalType.RULE_MATCH for s in signals)

    def test_question_mark_boost(self, generator: SignalGenerator) -> None:
        signals = _gen(generator, "你说什么？")
        rule_signals = [s for s in signals if s.signal_type == SignalType.RULE_MATCH]
        assert len(rule_signals) >= 1
        # 问号 + 关键词应该给较高分
        assert rule_signals[0].weight >= 0.3

    def test_english_question(self, generator: SignalGenerator) -> None:
        signals = _gen(generator, "how do you do that?")
        assert any(s.signal_type == SignalType.RULE_MATCH for s in signals)


class TestMentionDetection:
    """提及检测"""

    def test_mention_pattern(self, generator: SignalGenerator) -> None:
        signals = _gen(generator, "你怎么看这件事")
        rule_signals = [s for s in signals if s.signal_type == SignalType.RULE_MATCH]
        assert len(rule_signals) >= 1
        assert rule_signals[0].weight >= 0.4

    def test_help_pattern(self, generator: SignalGenerator) -> None:
        signals = _gen(generator, "帮我看看这个代码")
        rule_signals = [s for s in signals if s.signal_type == SignalType.RULE_MATCH]
        assert len(rule_signals) >= 1
        assert "mention" in rule_signals[0].metadata.get("matched_rules", [])


class TestAttentionDetection:
    """寻求关注检测"""

    def test_attention_keyword(self, generator: SignalGenerator) -> None:
        signals = _gen(generator, "有人吗，出来聊天")
        rule_signals = [s for s in signals if s.signal_type == SignalType.RULE_MATCH]
        assert len(rule_signals) >= 1
        assert "attention" in rule_signals[0].metadata.get("matched_rules", [])


class TestEmotionKeywordDetection:
    """情感关键词检测（规则层面）"""

    def test_negative_emotion(self, generator: SignalGenerator) -> None:
        signals = _gen(generator, "好烦啊，压力好大，快崩溃了")
        rule_signals = [s for s in signals if s.signal_type == SignalType.RULE_MATCH]
        assert len(rule_signals) >= 1
        rules = rule_signals[0].metadata.get("matched_rules", [])
        assert any("emotion_negative" in r for r in rules)

    def test_positive_emotion(self, generator: SignalGenerator) -> None:
        """正面情感词 + 寻求关注，组合达到阈值"""
        signals = _gen(generator, "太好了成功了！有人吗，好开心好激动")
        rule_signals = [s for s in signals if s.signal_type == SignalType.RULE_MATCH]
        assert len(rule_signals) >= 1
        rules = rule_signals[0].metadata.get("matched_rules", [])
        assert any("emotion_positive" in r for r in rules)


class TestEmotionHighDetection:
    """高情感信号检测（由外部情感强度触发）"""

    def test_low_intensity_no_signal(self, generator: SignalGenerator) -> None:
        signals = _gen(generator, "普通消息", emotion=0.3)
        emotion_signals = [s for s in signals if s.signal_type == SignalType.EMOTION_HIGH]
        assert emotion_signals == []

    def test_high_intensity_generates_signal(self, generator: SignalGenerator) -> None:
        signals = _gen(generator, "非常激动的消息", emotion=0.85)
        emotion_signals = [s for s in signals if s.signal_type == SignalType.EMOTION_HIGH]
        assert len(emotion_signals) == 1
        s = emotion_signals[0]
        assert s.weight >= 0.7
        assert s.metadata.get("emotion_intensity") == 0.85

    def test_boundary_intensity_0_7(self, generator: SignalGenerator) -> None:
        """刚好 0.7 的情感强度应触发 emotion_high 信号"""
        signals = _gen(generator, "有些激动", emotion=0.7)
        emotion_signals = [s for s in signals if s.signal_type == SignalType.EMOTION_HIGH]
        assert len(emotion_signals) == 1

    def test_below_boundary_no_signal(self, generator: SignalGenerator) -> None:
        signals = _gen(generator, "有些激动", emotion=0.69)
        emotion_signals = [s for s in signals if s.signal_type == SignalType.EMOTION_HIGH]
        assert emotion_signals == []


class TestCombinedSignals:
    """多重信号叠加测试"""

    def test_question_plus_emotion(self, generator: SignalGenerator) -> None:
        """同时触发规则匹配和高情感信号"""
        signals = _gen(generator, "你说什么？我好难过", emotion=0.8)
        types = {s.signal_type for s in signals}
        assert SignalType.RULE_MATCH in types
        assert SignalType.EMOTION_HIGH in types

    def test_no_signals_for_plain_text(self, generator: SignalGenerator) -> None:
        """普通文本不生成信号"""
        signals = _gen(generator, "今天天气不错")
        assert signals == []


class TestSignalMetadata:
    """信号元数据测试"""

    def test_text_preview_truncated(self, generator: SignalGenerator) -> None:
        long_text = "你说什么" + "A" * 100
        signals = _gen(generator, long_text)
        for s in signals:
            preview = s.metadata.get("text_preview", "")
            assert len(preview) <= 50

    def test_signal_has_expires_at(self, generator: SignalGenerator) -> None:
        signals = _gen(generator, "帮我看看这个问题")
        for s in signals:
            assert s.expires_at is not None

    def test_signal_session_key(self, generator: SignalGenerator) -> None:
        signals = _gen(generator, "帮我看看这个问题")
        for s in signals:
            assert s.session_key == "u1:g1"
            assert s.group_id == "g1"
            assert s.user_id == "u1"


class TestWeightRange:
    """权重范围测试"""

    def test_rule_weight_clamped(self, generator: SignalGenerator) -> None:
        """规则匹配权重应在 0.0 - 1.0"""
        signals = _gen(generator, "你怎么看这个什么问题？帮我看看")
        for s in signals:
            if s.signal_type == SignalType.RULE_MATCH:
                assert 0.0 <= s.weight <= 1.0

    def test_emotion_weight_clamped(self, generator: SignalGenerator) -> None:
        """情感权重应在 0.0 - 1.0"""
        signals = _gen(generator, "test", emotion=1.0)
        for s in signals:
            if s.signal_type == SignalType.EMOTION_HIGH:
                assert 0.0 <= s.weight <= 1.0

    def test_low_score_filtered(self, generator: SignalGenerator) -> None:
        """低于 0.2 的规则匹配不生成信号"""
        # "好" 可能不匹配任何高分关键词
        signals = _gen(generator, "早上好呀")
        # 这个可能不会触发（没有强关键词），也可能触发很低的分数
        for s in signals:
            if s.signal_type == SignalType.RULE_MATCH:
                assert s.weight >= 0.2
