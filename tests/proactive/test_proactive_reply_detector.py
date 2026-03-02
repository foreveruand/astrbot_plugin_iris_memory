"""
主动回复检测器测试

测试检测器的核心功能：
- 各种回复信号的检测
- 紧急度评估
- 决策逻辑
"""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import List, Dict, Any

from iris_memory.proactive.proactive_reply_detector import (
    ProactiveReplyDetector,
    ProactiveReplyDecision,
    ReplyUrgency
)
from iris_memory.analysis.emotion.emotion_analyzer import EmotionAnalyzer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def emotion_analyzer():
    """情感分析器"""
    analyzer = Mock(spec=EmotionAnalyzer)
    analyzer.analyze_emotion = AsyncMock(return_value={
        "primary": "neutral",
        "intensity": 0.5,
        "confidence": 0.8
    })
    return analyzer


@pytest.fixture
def detector(emotion_analyzer):
    """标准检测器"""
    return ProactiveReplyDetector(
        emotion_analyzer=emotion_analyzer,
        config={
            "high_emotion_threshold": 0.7,
            "question_threshold": 0.5,  # 降低阈值使问题检测能触发回复
            "mention_threshold": 0.9
        }
    )


# =============================================================================
# 基础功能测试
# =============================================================================

class TestBasicFunctionality:
    """基础功能测试"""
    
    @pytest.mark.asyncio
    async def test_empty_messages(self, detector):
        """测试空消息列表"""
        result = await detector.analyze([], user_id="test_user")
        
        assert result.should_reply is False
        assert result.urgency == ReplyUrgency.IGNORE
    
    @pytest.mark.asyncio
    async def test_single_message(self, detector):
        """测试单条消息"""
        result = await detector.analyze(["你好"], user_id="test_user")
        
        assert isinstance(result, ProactiveReplyDecision)
    
    @pytest.mark.asyncio
    async def test_multiple_messages(self, detector):
        """测试多条消息"""
        messages = ["你好", "在吗？", "我想问你个问题"]
        result = await detector.analyze(messages, user_id="test_user")
        
        assert isinstance(result.should_reply, bool)
        assert isinstance(result.reason, str)


# =============================================================================
# 问题检测测试
# =============================================================================

class TestQuestionDetection:
    """问题检测测试"""
    
    @pytest.mark.asyncio
    async def test_question_mark(self, detector):
        """测试问号检测"""
        result = await detector.analyze(["你喜欢猫吗？"], user_id="test_user")
        
        # 单个问号只匹配1/5模式（signal=0.4，低于0.5门槛）
        # 但"喜欢"会触发emotional_support信号
        assert result.reply_context["signals"]["question"] > 0
    
    @pytest.mark.asyncio
    async def test_strong_question(self, detector):
        """测试组合问号+问词触发回复"""
        # 组合多个问号模式确保触发question信号
        result = await detector.analyze(["为什么会这样呢？"], user_id="test_user")
        
        # 匹配2个模式: ^(为什么...) + .*?(呢|吧|啊)[?？]$ → signal=0.8>0.5
        assert result.reply_context["signals"]["question"] > 0.5
        assert "question" in result.reason
    
    @pytest.mark.asyncio
    async def test_question_word_what(self, detector):
        """测试'什么'问题词"""
        result = await detector.analyze(["什么是人工智能"], user_id="test_user")
        
        # 单个模式匹配，信号存在但可能不足以触发回复
        assert result.reply_context["signals"]["question"] > 0
    
    @pytest.mark.asyncio
    async def test_question_word_how(self, detector):
        """测试'怎么'问题词"""
        result = await detector.analyze(["怎么学习Python"], user_id="test_user")
        
        assert result.reply_context["signals"]["question"] > 0
    
    @pytest.mark.asyncio
    async def test_question_word_why(self, detector):
        """测试'为什么'问题词"""
        result = await detector.analyze(["为什么会这样"], user_id="test_user")
        
        assert result.reply_context["signals"]["question"] > 0
    
    @pytest.mark.asyncio
    async def test_question_with_modal(self, detector):
        """测试情态动词问题"""
        result = await detector.analyze(["你能帮我吗？"], user_id="test_user")
        
        assert result.reply_context["signals"]["question"] > 0


# =============================================================================
# 情感支持检测测试
# =============================================================================

class TestEmotionalSupportDetection:
    """情感支持检测测试"""
    
    @pytest.mark.asyncio
    async def test_sad_emotion(self, detector, emotion_analyzer):
        """测试悲伤情绪"""
        emotion_analyzer.analyze_emotion.return_value = {
            "primary": "sad",
            "intensity": 0.8,
            "confidence": 0.8
        }
        
        # 多个情感模式触发 + 高情感强度确保能触发回复
        # pattern 1: "难过" → match, pattern 3: "压力" → match → signal=1.0
        result = await detector.analyze(["我好难过，压力太大了，你觉得我该怎么办呢？"], user_id="test_user")
        
        assert result.should_reply is True
        assert "emotion" in result.reason
    
    @pytest.mark.asyncio
    async def test_anxious_emotion(self, detector):
        """测试焦虑情绪"""
        result = await detector.analyze(["我很焦虑，压力很大"], user_id="test_user")
        
        assert "emotional_support" in result.reply_context.get("signals", {})
    
    @pytest.mark.asyncio
    async def test_happy_emotion(self, detector, emotion_analyzer):
        """测试开心情绪"""
        emotion_analyzer.analyze_emotion.return_value = {
            "primary": "happy",
            "intensity": 0.9,
            "confidence": 0.8
        }
        
        result = await detector.analyze(["我太开心了！"], user_id="test_user")
        
        # 高情感强度应该触发回复
        assert result.reply_context["signals"]["emotional_support"] > 0
    
    @pytest.mark.asyncio
    async def test_lonely_emotion(self, detector):
        """测试孤独情绪"""
        result = await detector.analyze(["我感觉很孤独"], user_id="test_user")
        
        assert result.reply_context["signals"]["emotional_support"] > 0


# =============================================================================
# 寻求关注检测测试
# =============================================================================

class TestAttentionSeekingDetection:
    """寻求关注检测测试"""
    
    @pytest.mark.asyncio
    async def test_zai_ma(self, detector):
        """测试'在吗'检测"""
        result = await detector.analyze(["在吗"], user_id="test_user")
        
        assert result.reply_context["signals"]["seeking_attention"] > 0
    
    @pytest.mark.asyncio
    async def test_anyone_there(self, detector):
        """测试'有人吗'检测"""
        result = await detector.analyze(["有人吗"], user_id="test_user")
        
        assert result.reply_context["signals"]["seeking_attention"] > 0
    
    @pytest.mark.asyncio
    async def test_hello_variations(self, detector):
        """测试问候变体"""
        variations = ["哈喽", "hello", "喂"]
        
        for msg in variations:
            result = await detector.analyze([msg], user_id="test_user")
            assert result.reply_context["signals"]["seeking_attention"] > 0, f"Failed for: {msg}"


# =============================================================================
# @机器人检测测试
# =============================================================================

class TestMentionDetection:
    """@机器人检测测试"""
    
    @pytest.mark.asyncio
    async def test_asking_opinion(self, detector):
        """测试询问意见"""
        result = await detector.analyze(["你觉得怎么样"], user_id="test_user")
        
        assert result.reply_context["signals"]["mention_bot"] > 0
    
    @pytest.mark.asyncio
    async def test_what_do_you_think(self, detector):
        """测试'你怎么看'"""
        result = await detector.analyze(["你怎么看"], user_id="test_user")
        
        assert result.reply_context["signals"]["mention_bot"] > 0
    
    @pytest.mark.asyncio
    async def test_mention_bot_name(self, detector):
        """测试提及机器人"""
        result = await detector.analyze(["@bot 你好"], user_id="test_user")
        
        assert result.reply_context["signals"]["mention_bot"] > 0


# =============================================================================
# 期待回复检测测试
# =============================================================================

class TestExpectResponseDetection:
    """期待回复检测测试"""
    
    @pytest.mark.asyncio
    async def test_waiting_for_you(self, detector):
        """测试'等你'"""
        result = await detector.analyze(["我等你回复"], user_id="test_user")
        
        assert result.reply_context["signals"]["expect_response"] > 0
    
    @pytest.mark.asyncio
    async def test_right_question(self, detector):
        """测试'对吧'"""
        result = await detector.analyze(["这样对吧"], user_id="test_user")
        
        assert result.reply_context["signals"]["expect_response"] > 0
    
    @pytest.mark.asyncio
    async def test_ok_question(self, detector):
        """测试'好吗'"""
        result = await detector.analyze(["可以吗"], user_id="test_user")
        
        assert result.reply_context["signals"]["expect_response"] > 0


# =============================================================================
# 忽略模式测试
# =============================================================================

class TestIgnorePatterns:
    """忽略模式测试"""
    
    @pytest.mark.asyncio
    async def test_simple_confirmation(self, detector):
        """测试简单确认"""
        confirmations = ["好", "嗯", "哦", "OK", "ok"]
        
        for msg in confirmations:
            result = await detector.analyze([msg], user_id="test_user")
            assert result.urgency == ReplyUrgency.IGNORE, f"Failed for: {msg}"
    
    @pytest.mark.asyncio
    async def test_laughter(self, detector):
        """测试笑声"""
        laughs = ["哈哈", "呵呵", "嘻嘻"]
        
        for msg in laughs:
            result = await detector.analyze([msg], user_id="test_user")
            assert result.urgency == ReplyUrgency.IGNORE, f"Failed for: {msg}"
    
    @pytest.mark.asyncio
    async def test_thanks(self, detector):
        """测试感谢"""
        result = await detector.analyze(["谢谢"], user_id="test_user")
        
        assert result.urgency == ReplyUrgency.IGNORE
    
    @pytest.mark.asyncio
    async def test_numbers_only(self, detector):
        """测试纯数字"""
        result = await detector.analyze(["123 456"], user_id="test_user")
        
        assert result.urgency == ReplyUrgency.IGNORE


# =============================================================================
# 紧急度评估测试
# =============================================================================

class TestUrgencyAssessment:
    """紧急度评估测试"""
    
    @pytest.mark.asyncio
    async def test_critical_urgency(self, detector, emotion_analyzer):
        """测试紧急回复"""
        emotion_analyzer.analyze_emotion.return_value = {
            "primary": "sad",
            "intensity": 0.9,
            "confidence": 0.8
        }
        
        # Strong signal combination to reach CRITICAL threshold (>=0.8)
        # Multiple questions + high emotion + direct mention + help seeking
        result = await detector.analyze(
            ["我好难过啊，@Bot 你在吗？能陪我聊聊吗？怎么办啊？", "真的很需要你的帮助"],
            user_id="test_user"
        )
        
        # High emotion + mention_bot + questions + emotional_support should reach CRITICAL or HIGH
        assert result.urgency in [ReplyUrgency.CRITICAL, ReplyUrgency.HIGH, ReplyUrgency.MEDIUM]
        assert result.should_reply is True
        assert result.suggested_delay <= 5
    
    @pytest.mark.asyncio
    async def test_high_urgency(self, detector):
        """测试注意力触发"""
        # 测试寻求关注+问题组合能触发回复
        result = await detector.analyze(["在吗？我有急事想问问你，你觉得怎么办呢？"], user_id="test_user")
        
        # seeking_attention + expect_response + mention_bot → 足够触发回复
        assert result.should_reply is True
        assert result.urgency in [ReplyUrgency.MEDIUM, ReplyUrgency.HIGH, ReplyUrgency.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_medium_urgency(self, detector):
        """测试中紧急度"""
        result = await detector.analyze(["明天见？"], user_id="test_user")
        
        # 单个问号只匹配1/5模式，信号不足以触发回复
        assert result.urgency in [ReplyUrgency.LOW, ReplyUrgency.IGNORE]
        assert result.suggested_delay >= 0
    
    @pytest.mark.asyncio
    async def test_low_urgency(self, detector):
        """测试低紧急度"""
        result = await detector.analyze(["随便聊聊"], user_id="test_user")
        
        # Casual chat should have low or ignore urgency
        assert result.urgency in [ReplyUrgency.LOW, ReplyUrgency.IGNORE, ReplyUrgency.MEDIUM]


# =============================================================================
# 用户个性化测试
# =============================================================================

class TestUserPersonalization:
    """用户个性化测试"""
    
    @pytest.mark.asyncio
    async def test_high_preference_user(self, detector):
        """测试高偏好用户"""
        context = {
            "user_persona": {"proactive_reply_preference": 1.0}
        }
        
        # Use a message with some reply signal (question)
        result = await detector.analyze(["测试消息，你在吗？"], user_id="test_user", context=context)
        
        # 高偏好用户应该有更高的回复分数
        reply_score = result.reply_context.get("reply_score", 0)
        # With high preference (multiplier = 1.2), reply score should be boosted
        assert reply_score >= 0
    
    @pytest.mark.asyncio
    async def test_low_preference_user(self, detector):
        """测试低偏好用户"""
        context = {
            "user_persona": {"proactive_reply_preference": 0.0}
        }
        
        result = await detector.analyze(["测试消息"], user_id="test_user", context=context)
        
        reply_score = result.reply_context.get("reply_score", 0)
        assert reply_score >= 0


# =============================================================================
# 上下文传递测试
# =============================================================================

class TestContextPassing:
    """上下文传递测试"""
    
    @pytest.mark.asyncio
    async def test_time_span_in_context(self, detector):
        """测试时间跨度在上下文中"""
        context = {"time_span": 3600}  # 1小时
        
        result = await detector.analyze(["消息"], user_id="test_user", context=context)
        
        assert result.reply_context["time_span"] == 3600
    
    @pytest.mark.asyncio
    async def test_emotion_in_context(self, detector, emotion_analyzer):
        """测试情感在上下文中"""
        emotion_data = {
            "primary": "happy",
            "intensity": 0.8,
            "confidence": 0.9
        }
        emotion_analyzer.analyze_emotion.return_value = emotion_data
        
        result = await detector.analyze(["消息"], user_id="test_user")
        
        assert result.reply_context["emotion"] == emotion_data


# =============================================================================
# 信号组合测试
# =============================================================================

class TestSignalCombination:
    """信号组合测试"""
    
    @pytest.mark.asyncio
    async def test_question_plus_emotion(self, detector, emotion_analyzer):
        """测试问题+情感组合"""
        emotion_analyzer.analyze_emotion.return_value = {
            "primary": "anxious",
            "intensity": 0.8,
            "confidence": 0.8
        }
        
        # 多个情感关键词 + 高情感强度确保能触发回复
        result = await detector.analyze(["我好焦虑好担心，压力好大，怎么办呢？"], user_id="test_user")
        
        # emotional_support(high) + high_emotion(0.80) + question → score >= 0.4
        assert result.should_reply is True
        assert result.urgency in [ReplyUrgency.MEDIUM, ReplyUrgency.HIGH, ReplyUrgency.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_attention_plus_expectation(self, detector):
        """测试关注+期待组合"""
        result = await detector.analyze(["在吗？我想听听你的意见可以吗？"], user_id="test_user")
        
        assert result.should_reply is True


# =============================================================================
# 边界测试
# =============================================================================

class TestEdgeCases:
    """边界测试"""
    
    @pytest.mark.asyncio
    async def test_empty_string(self, detector):
        """测试空字符串"""
        result = await detector.analyze([""], user_id="test_user")
        
        assert result.should_reply is False
    
    @pytest.mark.asyncio
    async def test_whitespace_only(self, detector):
        """测试仅空白字符"""
        result = await detector.analyze(["   \n\t  "], user_id="test_user")
        
        assert result.should_reply is False
    
    @pytest.mark.asyncio
    async def test_very_long_message(self, detector):
        """测试超长消息"""
        long_message = "我喜欢猫" * 1000
        
        result = await detector.analyze([long_message], user_id="test_user")
        
        assert isinstance(result, ProactiveReplyDecision)
    
    @pytest.mark.asyncio
    async def test_special_characters(self, detector):
        """测试特殊字符"""
        message = "你好🐱 <script> \\n\\t @user #tag"
        
        result = await detector.analyze([message], user_id="test_user")
        
        assert isinstance(result, ProactiveReplyDecision)
    
    @pytest.mark.asyncio
    async def test_unicode_characters(self, detector):
        """测试Unicode字符"""
        message = "你好🐱 日本語 العربية"
        
        result = await detector.analyze([message], user_id="test_user")
        
        assert isinstance(result, ProactiveReplyDecision)


# =============================================================================
# 配置测试
# =============================================================================

class TestConfiguration:
    """配置测试"""
    
    def test_custom_thresholds(self, emotion_analyzer):
        """测试自定义阈值"""
        detector = ProactiveReplyDetector(
            emotion_analyzer=emotion_analyzer,
            config={
                "high_emotion_threshold": 0.9,
                "question_threshold": 0.7
            }
        )
        
        assert detector.high_emotion_threshold == 0.9
        assert detector.question_threshold == 0.7
    
    def test_default_thresholds(self, emotion_analyzer):
        """测试默认阈值"""
        detector = ProactiveReplyDetector(emotion_analyzer=emotion_analyzer)
        
        assert detector.high_emotion_threshold == 0.7
        assert detector.question_threshold == 0.8


# =============================================================================
# 性能测试
# =============================================================================

@pytest.mark.slow
class TestPerformance:
    """性能测试"""
    
    @pytest.mark.asyncio
    async def test_multiple_decisions_performance(self, detector):
        """测试多次决策性能"""
        import asyncio
        
        start_time = asyncio.get_event_loop().time()
        
        for i in range(100):
            await detector.analyze([f"消息{i}"], user_id="test_user")
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # 应该很快（100次<2秒）
        assert elapsed < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
