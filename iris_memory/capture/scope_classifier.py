"""
记忆可见性智能分类器

LLM+规则双层架构：强隐私规则优先 → 敏感度检查 → 个人模式 → 群公告规则 → LLM 分类 → 默认 GROUP_SHARED。
群聊默认共享，根据规则降级为私有。
"""

import re
from typing import Any, ClassVar

from iris_memory.core.memory_scope import MemoryScope
from iris_memory.core.types import SensitivityLevel
from iris_memory.utils.logger import get_logger

logger = get_logger("scope_classifier")


class ScopeClassifier:
    """记忆可见性智能分类器"""

    # 强隐私关键词 → USER_PRIVATE
    STRONG_PRIVATE_PATTERNS: ClassVar[list[str]] = [
        r"别告诉(?:别人|其他人|他们|任何人)",
        r"(?:这是|有个)秘密",
        r"不要(?:跟|和).*说",
        r"悄悄(?:告诉你|说|地)",
        r"别外传",
        r"我(?:的)?(?:密码|账号|银行卡|身份证号)",
    ]

    # 群公告关键词 → GROUP_SHARED
    GROUP_SHARED_PATTERNS: ClassVar[list[str]] = [
        r"(?:群|大家)(?:规|约定|公告|通知)",
        r"(?:所有人|各位|大家)(?:注意|看这里)",
        r"@(?:全体成员|所有人|all|everyone)",
    ]

    # 个人信息关键词 → GROUP_PRIVATE
    PERSONAL_PATTERNS: ClassVar[list[str]] = [
        r"^我(?:是|在|有|喜欢|讨厌|觉得|认为|想|要|不)",
        r"我(?:自己|个人|一个人)",
        r"我的(?:名字|工作|家|手机|电脑|生日|爱好|习惯)",
    ]

    LLM_SCOPE_PROMPT: ClassVar[str] = (
        "分析以下消息在群聊环境中的记忆可见性。\n\n"
        '消息: "{message}"\n发送者: {sender}\n群组: {group_id}\n\n'
        "判断该消息产生的记忆应该属于哪种可见性范围:\n"
        "- USER_PRIVATE: 用户个人隐私，仅用户自己可见\n"
        "- GROUP_PRIVATE: 群内个人记忆，仅发送者在该群可见\n"
        "- GROUP_SHARED: 群共享信息，群内所有人可见（默认）\n\n"
        '返回JSON: {{"scope": "...", "confidence": 0.0-1.0, "reason": "..."}}'
    )

    def __init__(self, llm_provider=None):
        self._llm = llm_provider

    async def classify(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> MemoryScope:
        """智能分类记忆可见性

        群聊默认 GROUP_SHARED，根据规则降级为私有。
        """
        if context is None:
            context = {}

        is_group = bool(context.get("is_group") or context.get("group_id"))

        if not is_group:
            return MemoryScope.USER_PRIVATE

        for pattern in self.STRONG_PRIVATE_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return MemoryScope.USER_PRIVATE

        sensitivity = context.get("sensitivity_level")
        if sensitivity is not None:
            sens_val = (
                sensitivity.value
                if isinstance(sensitivity, SensitivityLevel)
                else int(sensitivity)
            )
            if sens_val >= SensitivityLevel.PRIVATE.value:
                return MemoryScope.USER_PRIVATE

        for pattern in self.PERSONAL_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return MemoryScope.GROUP_PRIVATE

        for pattern in self.GROUP_SHARED_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return MemoryScope.GROUP_SHARED

        if self._llm and is_group:
            try:
                llm_result = await self._llm_classify(message, context)
                if llm_result and llm_result.get("confidence", 0) > 0.8:
                    scope_str = llm_result["scope"].lower()
                    return MemoryScope(scope_str)
            except Exception:
                pass

        return MemoryScope.GROUP_SHARED

    async def _llm_classify(
        self, message: str, context: dict[str, Any]
    ) -> dict[str, Any] | None:
        """使用 LLM 分类可见性"""
        import json

        prompt = self.LLM_SCOPE_PROMPT.format(
            message=message[:200],
            sender=context.get("sender", "unknown"),
            group_id=context.get("group_id", "unknown"),
        )
        try:
            response = await self._llm(prompt)
            if response:
                return json.loads(response)
        except (json.JSONDecodeError, TypeError):
            pass
        return None
