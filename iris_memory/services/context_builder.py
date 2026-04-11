"""LLM 上下文构建器

从 BusinessService 中提取的 prepare_llm_context 及其辅助方法，
负责将记忆、聊天记录、画像、知识图谱、图片描述等拼装为 LLM prompt。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from iris_memory.core.constants import LogTemplates, PersonaStyle
from iris_memory.persona.persona_logger import persona_log
from iris_memory.utils.command_utils import SessionKeyBuilder
from iris_memory.utils.logger import get_logger
from iris_memory.utils.member_utils import format_member_tag

if TYPE_CHECKING:
    from iris_memory.services.modules.analysis_module import AnalysisModule
    from iris_memory.services.modules.kg_module import KnowledgeGraphModule
    from iris_memory.services.modules.retrieval_module import RetrievalModule
    from iris_memory.services.modules.storage_module import StorageModule
    from iris_memory.services.shared_state import SharedState

logger = get_logger("memory_service.context_builder")


@dataclass(frozen=True)
class PromptSection:
    """单个提示词注入分段。"""

    category: str
    content: str


class ContextBuilder:
    """LLM 上下文构建器

    负责将多源信息组装为完整的 LLM prompt 上下文字符串。

    依赖：
    - retrieval: 记忆检索引擎
    - analysis: 情感分析器
    - storage: 聊天记录缓冲区
    - kg: 知识图谱上下文
    - shared_state: 用户画像和情感状态
    - cfg: 配置管理器
    """

    def __init__(
        self,
        retrieval: RetrievalModule,
        analysis: AnalysisModule,
        storage: StorageModule,
        kg: KnowledgeGraphModule,
        shared_state: SharedState,
        cfg: Any,
        member_identity: Any = None,
    ) -> None:
        self._retrieval = retrieval
        self._analysis = analysis
        self._storage = storage
        self._kg = kg
        self._state = shared_state
        self._cfg = cfg
        self._member_identity = member_identity

    def set_member_identity(self, identity: Any) -> None:
        """更新成员身份服务引用"""
        self._member_identity = identity

    # ── 主入口 ──

    async def build(
        self,
        query: str,
        user_id: str,
        group_id: str | None,
        image_context: str = "",
        sender_name: str | None = None,
        reply_context: str | None = None,
        persona_id: str | None = None,
    ) -> str:
        """构建完整的 LLM 上下文字符串（兼容旧接口）。"""
        sections = await self.build_sections(
            query=query,
            user_id=user_id,
            group_id=group_id,
            image_context=image_context,
            sender_name=sender_name,
            reply_context=reply_context,
            persona_id=persona_id,
        )
        return self._compose_sections_for_prompt(sections)

    async def build_sections(
        self,
        query: str,
        user_id: str,
        group_id: str | None,
        image_context: str = "",
        sender_name: str | None = None,
        reply_context: str | None = None,
        persona_id: str | None = None,
    ) -> list[PromptSection]:
        """构建结构化提示词分段，供不同注入方式复用。"""
        if not self._retrieval.retrieval_engine:
            return []

        try:
            emotional_state = self._state.get_or_create_emotional_state(user_id)
            if self._analysis.emotion_analyzer:
                emotion_result = await self._analysis.emotion_analyzer.analyze_emotion(
                    query
                )
                self._analysis.emotion_analyzer.update_emotional_state(
                    emotional_state,
                    emotion_result["primary"],
                    emotion_result["intensity"],
                    emotion_result["confidence"],
                    emotion_result["secondary"],
                )

            memories = await self._retrieval.retrieval_engine.retrieve(
                query=query,
                user_id=user_id,
                group_id=group_id,
                top_k=self._cfg.get("memory.max_context_memories", 10),
                emotional_state=emotional_state,
                persona_id=persona_id,
            )

            session_key = SessionKeyBuilder.build(user_id, group_id)
            if memories:
                memories = self._state.filter_recently_injected(memories, session_key)

            sections: list[PromptSection] = []

            chat_context = await self._build_chat_history(user_id, group_id)
            if chat_context:
                sections.append(PromptSection("chat_history", chat_context))

            persona = self._state.get_or_create_user_persona(user_id)
            persona_view: dict[str, Any] | None = (
                persona.to_injection_view() if persona else None
            )
            if persona_view:
                persona_log.inject_view(user_id, persona_view)

            if memories:
                memory_context = (
                    self._retrieval.retrieval_engine.format_memories_for_llm(
                        memories,
                        persona_style=PersonaStyle.NATURAL,
                        user_persona=None,
                        group_id=group_id,
                        current_sender_name=sender_name,
                    )
                )
                if memory_context:
                    sections.append(PromptSection("memory", memory_context))
                    logger.debug(
                        LogTemplates.MEMORY_INJECTED.format(count=len(memories))
                    )

                self._state.track_injected_memories(
                    session_key, [m.id for m in memories]
                )

            if persona_view:
                persona_context = (
                    self._retrieval.retrieval_engine.format_persona_context_for_llm(
                        user_persona=persona_view,
                        bot_persona="friendly",
                    )
                )
                if persona_context:
                    sections.append(PromptSection("persona", persona_context))

            member_context = self._build_member_identity(
                memories, group_id, user_id, sender_name
            )
            if member_context:
                sections.append(PromptSection("behavior", member_context))

            # 知识图谱上下文
            if self._kg and self._kg.enabled:
                try:
                    kg_context = await self._kg.format_graph_context(
                        query=query,
                        user_id=user_id,
                        group_id=group_id,
                        persona_id=persona_id,
                    )
                    if kg_context:
                        sections.append(PromptSection("knowledge_graph", kg_context))
                        logger.debug("Injected knowledge graph context into LLM prompt")
                except Exception as kg_err:
                    logger.debug(f"KG context skipped: {kg_err}")

            if image_context:
                sections.append(PromptSection("image_context", image_context))
                logger.debug("Injected image context into LLM prompt")

            if reply_context:
                sections.append(PromptSection("reply_context", reply_context))
                logger.debug("Injected reply context into LLM prompt")

            behavior_directives = self._build_behavior_directives(group_id, sender_name)
            if behavior_directives:
                sections.append(PromptSection("behavior", behavior_directives))

            return sections

        except Exception as e:
            logger.warning(f"Failed to prepare LLM context: {e}")
            return []

    def _compose_sections_for_prompt(self, sections: list[PromptSection]) -> str:
        """根据各类别的注入位置组合为单个 prompt 字符串。"""
        prepend_parts: list[str] = []
        append_parts: list[str] = []

        for section in sections:
            if not section.content:
                continue
            position = self._cfg.get(
                f"prompt_injection.{section.category}.position", "append"
            )
            if position == "prepend":
                prepend_parts.append(section.content)
            else:
                append_parts.append(section.content)

        ordered_parts = [*prepend_parts, *append_parts]
        return "\n\n".join(part for part in ordered_parts if part)

    # ── 辅助构建方法 ──

    def _build_behavior_directives(
        self, group_id: str | None, sender_name: str | None = None
    ) -> str:
        """构建记忆参考指导，仅涉及如何引用和查询记忆，不干预回复风格。"""
        directives = [
            "【记忆参考指南】",
            "◆ 参考而非宣告：注入的记忆仅供参考，无需主动逐条引用或展示，将其融合进对话背景理解即可。",
            "◆ 置信度权重：置信度较低（< 0.3）的记忆可能不准确，参考时保持适当存疑，不要将其当作确定事实断言。",
            "◆ 额外检索：若当前注入的记忆不足以回答用户的问题，或用户明确询问你是否记得某件事，请调用 search_memory 工具主动检索更多相关记忆，再作回答。",
        ]

        if group_id:
            directives.append(
                "◆ 归属区分：记忆中标注了「群聊共识」和「个人信息」。引用个人信息时须确认归属当前对话者，不要将其他群成员的信息错误地归到当前用户身上。"
            )
        else:
            directives.append(
                "◆ 归属确认：这是私聊对话，注入的记忆均属于当前对话的双方。"
            )

        return "\n".join(directives)

    def _build_member_identity(
        self,
        memories: list[Any],
        group_id: str | None,
        user_id: str,
        sender_name: str | None,
    ) -> str:
        """Build a compact member identity hint for group chats."""
        if not group_id:
            return ""

        current_tag = format_member_tag(sender_name, user_id, group_id)
        other_tags: list[str] = []
        seen: set = set()

        for memory in memories:
            tag = format_member_tag(memory.sender_name, memory.user_id, group_id)
            if not tag:
                continue
            if tag == current_tag:
                continue
            if tag in seen:
                continue
            seen.add(tag)
            other_tags.append(tag)

        lines = [
            "【群成员识别】",
            f"当前对话者: {current_tag}。回复时针对这个人，不要混淆成其他群友。",
        ]

        if other_tags:
            lines.append("记忆中涉及成员: " + ", ".join(other_tags[:5]))

        if self._member_identity:
            all_members = self._member_identity.get_group_members(group_id)
            extra_members = [
                m for m in all_members if m != current_tag and m not in seen
            ]
            if extra_members:
                lines.append("群内其他已知成员: " + ", ".join(extra_members[:10]))

            history = self._member_identity.get_name_history(user_id)
            if history:
                last_change = history[-1]
                lines.append(
                    f'注意: 当前对话者曾用名 "{last_change["old_name"]}"，'
                    f'现在叫 "{last_change["new_name"]}"。'
                )

        lines.append(
            "同名以#后ID区分。不要把A说的话当成B说的，引用其他人的记忆时要明确说明。"
        )

        return "\n".join(lines)

    async def _build_chat_history(self, user_id: str, group_id: str | None) -> str:
        """构建聊天记录上下文

        根据配置开关决定是否注入：
        - 群聊场景：检查 chat_history.enable_group
        - 私聊场景：检查 chat_history.enable_private
        """
        # 检查是否启用聊天历史注入
        if group_id:
            # 群聊场景
            if not self._cfg.get("chat_history.enable_group", True):
                return ""
        else:
            # 私聊场景
            if not self._cfg.get("chat_history.enable_private", True):
                return ""

        chat_history_buffer = self._storage.chat_history_buffer
        if not chat_history_buffer:
            return ""

        # 优先使用活动自适应配置（ActivityConfig.get_chat_context_count）
        chat_context_getter = getattr(self._cfg, "get_chat_context_count", None)
        if callable(chat_context_getter) and (
            not hasattr(self._cfg, "__dict__")
            or "get_chat_context_count" in self._cfg.__dict__
        ):
            chat_context_count = chat_context_getter(group_id)
        else:
            chat_context_count = self._cfg.get("advanced.chat_context_count", 15)

        if chat_context_count <= 0:
            return ""

        if chat_history_buffer.max_messages < chat_context_count:
            chat_history_buffer.set_max_messages(chat_context_count)

        messages = await chat_history_buffer.get_recent_messages(
            user_id=user_id, group_id=group_id, limit=chat_context_count
        )

        if not messages:
            return ""

        context = chat_history_buffer.format_for_llm(messages, group_id=group_id)

        if context:
            logger.debug(
                f"Injected {len(messages)} chat messages into context "
                f"(group={group_id is not None})"
            )

        return context
