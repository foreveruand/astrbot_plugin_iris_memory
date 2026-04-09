"""
知识图谱三元组提取器

从消息文本中提取 (主语, 谓语, 宾语) 三元组，并将其映射到图节点和边。

支持两种模式：
- rule : 纯规则/正则提取（零 LLM 开销）
- llm  : 使用 LLM 进行语义级关系提取（精度高）
- hybrid: 规则预筛 + LLM 补充（推荐）
"""

from __future__ import annotations

from typing import Any

from iris_memory.knowledge_graph.kg_models import (
    KGEdge,
    KGNode,
    KGNodeType,
    KGRelationType,
    KGTriple,
)
from iris_memory.knowledge_graph.kg_patterns import (
    BATCH_TRIPLE_EXTRACTION_PROMPT as _BATCH_TRIPLE_EXTRACTION_PROMPT,
)
from iris_memory.knowledge_graph.kg_patterns import (
    CN_RELATION_PATTERNS as _CN_RELATION_PATTERNS,
)
from iris_memory.knowledge_graph.kg_patterns import (
    DEFAULT_DAILY_LIMIT as _DEFAULT_DAILY_LIMIT,
)
from iris_memory.knowledge_graph.kg_patterns import (
    EN_RELATION_PATTERNS as _EN_RELATION_PATTERNS,
)
from iris_memory.knowledge_graph.kg_patterns import (
    QUICK_FILTER_KEYWORDS as _QUICK_FILTER_KEYWORDS,
)
from iris_memory.knowledge_graph.kg_patterns import (
    RELATIONSHIP_SIGNAL_KEYWORDS as _RELATIONSHIP_SIGNAL_KEYWORDS,
)
from iris_memory.knowledge_graph.kg_patterns import (
    RULE_TEXT_MAX_LENGTH,
)
from iris_memory.knowledge_graph.kg_patterns import (
    TRIPLE_EXTRACTION_PROMPT as _TRIPLE_EXTRACTION_PROMPT,
)
from iris_memory.knowledge_graph.kg_patterns import (
    guess_node_type as _guess_node_type,
)
from iris_memory.knowledge_graph.kg_storage import KGStorage
from iris_memory.utils.logger import get_logger
from iris_memory.utils.rate_limiter import DailyCallLimiter

logger = get_logger("kg_extractor")


class KGExtractor:
    """三元组提取器

    从文本中提取实体关系，并写入 KGStorage。

    支持三种模式：
    - rule: 正则/规则提取
    - llm: LLM 语义提取
    - hybrid: 规则 + LLM
    """

    def __init__(
        self,
        storage: KGStorage,
        mode: str = "rule",
        astrbot_context: Any = None,
        provider_id: str | None = None,
        daily_limit: int = _DEFAULT_DAILY_LIMIT,
    ) -> None:
        self.storage = storage
        self.mode = mode  # "rule" | "llm" | "hybrid"
        self._astrbot_context = astrbot_context
        self._provider_id = provider_id
        self._provider = None
        self._resolved_provider_id: str | None = None
        self._provider_initialized = False

        # 每日 LLM 调用限制
        self._limiter = DailyCallLimiter(daily_limit)

        # Hybrid 决策统计
        self._stats: dict[str, int] = {
            "rule_extractions": 0,
            "llm_extractions": 0,
            "llm_skipped_sufficient": 0,
            "llm_skipped_limit": 0,
            "llm_skipped_no_signal": 0,
            "hybrid_decisions": 0,
            "total_triples": 0,
        }

    # ================================================================
    # 主入口
    # ================================================================

    async def extract_and_store(
        self,
        text: str,
        user_id: str,
        group_id: str | None = None,
        memory_id: str | None = None,
        sender_name: str | None = None,
        existing_entities: list[str] | None = None,
        persona_id: str | None = None,
    ) -> list[KGTriple]:
        """从文本中提取三元组并存入图谱

        Args:
            text: 消息文本
            user_id: 用户 ID
            group_id: 群组 ID
            memory_id: 关联的记忆 ID
            sender_name: 发送者名称
            existing_entities: 已提取的实体列表（来自 EntityExtractor）
            persona_id: 人格 ID（始终写入节点/边，用于 persona 隔离）

        Returns:
            提取到的三元组列表
        """
        if not text or len(text.strip()) < 4:
            return []

        triples: list[KGTriple] = []

        if self.mode in ("rule", "hybrid"):
            rule_triples = self._extract_by_rules(text, sender_name)
            triples.extend(rule_triples)
            if rule_triples:
                self._stats["rule_extractions"] += 1

        if self.mode == "llm":
            # 纯 LLM 模式：直接调用
            llm_triples = await self._extract_by_llm(text, user_id, sender_name)
            triples = self._merge_triples(triples, llm_triples)
            if llm_triples:
                self._stats["llm_extractions"] += 1
        elif self.mode == "hybrid":
            # hybrid 模式：条件触发 LLM
            self._stats["hybrid_decisions"] += 1
            should_call, reason = self._should_call_llm_hybrid(text, triples)
            if should_call:
                if self._limiter.is_within_limit():
                    llm_triples = await self._extract_by_llm(text, user_id, sender_name)
                    if llm_triples:
                        self._limiter.increment()
                        self._stats["llm_extractions"] += 1
                    triples = self._merge_triples(triples, llm_triples)
                    logger.debug(
                        f"Hybrid LLM triggered ({reason}): "
                        f"got {len(llm_triples) if llm_triples else 0} triples"
                    )
                else:
                    self._stats["llm_skipped_limit"] += 1
                    logger.debug("Hybrid LLM skipped: daily limit reached")
            else:
                logger.debug(f"Hybrid LLM skipped: {reason}")

        if not triples:
            # 尝试从 existing_entities 构建隐含关系
            if existing_entities and sender_name:
                triples = self._build_implicit_triples(
                    text, sender_name, existing_entities
                )

        # 写入 KGStorage
        for triple in triples:
            await self._store_triple(triple, user_id, group_id, memory_id, persona_id)

        if triples:
            self._stats["total_triples"] += len(triples)
            logger.debug(
                f"Extracted {len(triples)} triples from text (mode={self.mode}): "
                + ", ".join(str(t) for t in triples[:3])
            )

        return triples

    # ================================================================
    # Hybrid 决策逻辑
    # ================================================================

    def _should_call_llm_hybrid(
        self,
        text: str,
        rule_triples: list[KGTriple],
    ) -> tuple[bool, str]:
        """判断 hybrid 模式下是否需要调用 LLM

        决策逻辑：
        1. 规则已提取到 ≥2 个高置信度三元组 → 跳过
        2. 文本超长（规则跳过了）→ 触发 LLM
        3. 文本含有关系信号但规则未提取到 → 触发 LLM
        4. 规则只提取到低置信度结果 → 触发 LLM 补充

        Returns:
            (should_call, reason) 元组
        """
        # 规则已提取到足够多的高置信度结果
        high_conf = [t for t in rule_triples if t.confidence >= 0.6]
        if len(high_conf) >= 2:
            self._stats["llm_skipped_sufficient"] += 1
            return False, "rule_sufficient"

        # 文本超长（规则因长度限制跳过了）
        if len(text) > RULE_TEXT_MAX_LENGTH:
            return True, "text_too_long_for_rules"

        # 规则未提取到但文本有关系信号
        if not rule_triples and self._has_relationship_signals(text):
            return True, "relationship_signals_detected"

        # 规则提取到了但全部低置信度
        if rule_triples and all(t.confidence < 0.5 for t in rule_triples):
            return True, "low_confidence_rules"

        # 规则提取到 1 个高置信度结果，但文本可能还有更多关系
        if len(high_conf) == 1 and self._has_relationship_signals(text):
            return True, "partial_rules_with_signals"

        # 其余情况不调用
        if not rule_triples:
            self._stats["llm_skipped_no_signal"] += 1
            return False, "no_signal"

        self._stats["llm_skipped_sufficient"] += 1
        return False, "rule_acceptable"

    @staticmethod
    def _has_relationship_signals(text: str) -> bool:
        """检测文本是否包含关系描述的信号词

        除了 _QUICK_FILTER_KEYWORDS（用于规则匹配的精确关键词），
        这里额外覆盖一些规则正则无法匹配但 LLM 能理解的模式。
        """
        # 先检查规则关键词（这些文本至少可能包含关系）
        if any(kw in text for kw in _QUICK_FILTER_KEYWORDS):
            return True
        # 再检查扩展信号词
        if any(kw in text for kw in _RELATIONSHIP_SIGNAL_KEYWORDS):
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """获取 Hybrid 决策统计"""
        return {
            **self._stats,
            "daily_limit": self._limiter._daily_limit,
            "remaining_calls": self._limiter.remaining,
            "mode": self.mode,
        }

    @property
    def remaining_daily_calls(self) -> int:
        """剩余每日 LLM 可用次数"""
        return self._limiter.remaining

    # ================================================================
    # 规则提取
    # ================================================================

    def _extract_by_rules(
        self,
        text: str,
        sender_name: str | None = None,
    ) -> list[KGTriple]:
        """基于预编译正则模式提取三元组

        优化策略：
        1. 文本长度阈值：超长文本跳过规则提取
        2. 关键词预过滤：快速检测文本是否可能包含关系表述
        3. 预编译正则：避免重复编译开销
        """
        # 超长文本跳过规则提取（交给 LLM 处理）
        if len(text) > RULE_TEXT_MAX_LENGTH:
            logger.debug(
                f"Text too long ({len(text)} chars > {RULE_TEXT_MAX_LENGTH}), "
                f"skipping rule extraction"
            )
            return []

        # 快速关键词预过滤
        if not any(kw in text for kw in _QUICK_FILTER_KEYWORDS):
            return []

        triples: list[KGTriple] = []
        seen: set = set()

        all_patterns = _CN_RELATION_PATTERNS + _EN_RELATION_PATTERNS

        for compiled_pattern, relation_type, label in all_patterns:
            for m in compiled_pattern.finditer(text):
                subject = m.group("s").strip()
                obj = m.group("o").strip()

                # 替换代词
                subject = self._resolve_pronoun(subject, sender_name)
                obj = self._resolve_pronoun(obj, sender_name)

                if not subject or not obj or subject == obj:
                    continue

                key = (subject.lower(), relation_type.value, obj.lower())
                if key in seen:
                    continue
                seen.add(key)

                triple = KGTriple(
                    subject=subject,
                    predicate=label,
                    object=obj,
                    subject_type=_guess_node_type(subject),
                    object_type=_guess_node_type(obj),
                    relation_type=relation_type,
                    confidence=0.7,
                    source_text=text,
                )
                triples.append(triple)

        return triples

    def _resolve_pronoun(self, text: str, sender_name: str | None) -> str:
        """将代词替换为发送者名称"""
        pronouns_cn = {"我", "本人", "自己", "俺", "吾"}
        pronouns_en = {"i", "me", "myself"}

        if text in pronouns_cn or text.lower() in pronouns_en:
            return sender_name or text
        return text

    def _build_implicit_triples(
        self,
        text: str,
        sender_name: str,
        entities: list[str],
    ) -> list[KGTriple]:
        """从已提取实体构建隐含关系"""
        triples: list[KGTriple] = []

        for entity in entities:
            if entity == sender_name:
                continue
            # 默认创建 "related_to" 关系
            triple = KGTriple(
                subject=sender_name,
                predicate="提到了",
                object=entity,
                subject_type=KGNodeType.PERSON,
                object_type=_guess_node_type(entity),
                relation_type=KGRelationType.RELATED_TO,
                confidence=0.3,
                source_text=text,
            )
            triples.append(triple)

        return triples[:3]  # 限制隐含关系数量

    # ================================================================
    # LLM 提取
    # ================================================================

    async def _extract_by_llm(
        self,
        text: str,
        user_id: str,
        sender_name: str | None,
    ) -> list[KGTriple]:
        """使用 LLM 提取三元组"""
        if not self._astrbot_context:
            return []

        try:
            if not await self._ensure_provider():
                return []

            prompt = _TRIPLE_EXTRACTION_PROMPT.format(
                text=text,
                sender_name=sender_name or "未知",
                user_id=user_id,
            )

            from iris_memory.utils.llm_helper import call_llm, parse_llm_json

            result = await call_llm(
                self._astrbot_context,
                self._provider,
                self._resolved_provider_id,
                prompt,
                parse_json=True,
            )

            if not result.success or not result.content:
                return []

            data = result.parsed_json or parse_llm_json(result.content)
            if not data or "triples" not in data:
                return []

            triples: list[KGTriple] = []
            for item in data["triples"][:5]:
                triple = self._parse_single_triple(item, text)
                if triple:
                    triples.append(triple)

            return triples

        except Exception as e:
            logger.warning(f"LLM triple extraction failed: {e}")
            return []

    async def _ensure_provider(self) -> bool:
        """确保 LLM provider 可用

        采用延迟初始化策略：仅在成功解析后缓存结果。
        AstrBot 先加载插件后加载 provider，首次调用时 provider 可能尚不可用，
        因此失败时不设置 _provider_initialized，允许后续调用重试。
        """
        if self._provider_initialized:
            return self._provider is not None

        try:
            from iris_memory.utils.llm_helper import resolve_llm_provider

            provider, resolved_provider_id = resolve_llm_provider(
                self._astrbot_context,
                self._provider_id or "",
                label="KGExtractor",
            )
            if provider:
                self._provider = provider
                self._resolved_provider_id = resolved_provider_id
                self._provider_initialized = True
                return True
        except Exception as e:
            logger.warning(f"Failed to resolve LLM provider for KGExtractor: {e}")

        # 不设置 _provider_initialized = True，允许后续重试
        return False

    # ================================================================
    # 三元组 → 图节点/边
    # ================================================================

    async def _store_triple(
        self,
        triple: KGTriple,
        user_id: str,
        group_id: str | None,
        memory_id: str | None,
        persona_id: str | None = None,
    ) -> None:
        """将三元组写入 KGStorage"""
        _persona = persona_id or "default"
        # 创建/更新主语节点
        subject_node = KGNode(
            name=triple.subject,
            display_name=triple.subject,
            node_type=triple.subject_type,
            user_id=user_id,
            group_id=group_id,
            persona_id=_persona,
            confidence=triple.confidence,
        )
        subject_node = await self.storage.upsert_node(subject_node)

        # 创建/更新宾语节点
        object_node = KGNode(
            name=triple.object,
            display_name=triple.object,
            node_type=triple.object_type,
            user_id=user_id,
            group_id=group_id,
            persona_id=_persona,
            confidence=triple.confidence,
        )
        object_node = await self.storage.upsert_node(object_node)

        # 创建/更新边
        edge = KGEdge(
            source_id=subject_node.id,
            target_id=object_node.id,
            relation_type=triple.relation_type,
            relation_label=triple.predicate,
            memory_id=memory_id,
            user_id=user_id,
            group_id=group_id,
            persona_id=_persona,
            confidence=triple.confidence,
        )
        await self.storage.upsert_edge(edge)

    # ================================================================
    # 工具方法
    # ================================================================

    def _merge_triples(
        self,
        rule_triples: list[KGTriple],
        llm_triples: list[KGTriple],
    ) -> list[KGTriple]:
        """合并规则和 LLM 提取的三元组（去重）"""
        seen: set = set()
        merged: list[KGTriple] = []

        for t in rule_triples:
            key = (t.subject.lower(), t.relation_type.value, t.object.lower())
            seen.add(key)
            merged.append(t)

        for t in llm_triples:
            key = (t.subject.lower(), t.relation_type.value, t.object.lower())
            if key not in seen:
                seen.add(key)
                merged.append(t)

        return merged

    # ================================================================
    # 批量 LLM 提取
    # ================================================================

    async def batch_extract_by_llm(
        self,
        items: list[dict[str, Any]],
    ) -> dict[int, list[KGTriple]]:
        """批量提取：将多条消息合并为 1 次 LLM 调用

        Args:
            items: 列表，每项包含 text, user_id, sender_name, index（原始索引）

        Returns:
            { index: [KGTriple, ...] } 映射
        """
        if not items or not self._astrbot_context:
            return {}

        try:
            if not await self._ensure_provider():
                return {}

            # 构建多消息 prompt
            messages_parts: list[str] = []
            for i, item in enumerate(items):
                sender = item.get("sender_name") or "未知"
                user_id = item.get("user_id", "")
                messages_parts.append(
                    f"### 消息{i} (发送者: {sender}, 用户ID: {user_id})\n{item['text']}"
                )
            messages_text = "\n\n".join(messages_parts)

            prompt = _BATCH_TRIPLE_EXTRACTION_PROMPT.format(
                messages_text=messages_text,
                message_count=len(items),
            )

            from iris_memory.utils.llm_helper import call_llm, parse_llm_json

            result = await call_llm(
                self._astrbot_context,
                self._provider,
                self._resolved_provider_id,
                prompt,
                parse_json=True,
            )

            if not result.success or not result.content:
                return {}

            data = result.parsed_json or parse_llm_json(result.content)
            return self._parse_batch_llm_response(data, items)

        except Exception as e:
            logger.warning(f"Batch LLM triple extraction failed: {e}")
            return {}

    def _parse_batch_llm_response(
        self,
        data: Any,
        items: list[dict[str, Any]],
    ) -> dict[int, list[KGTriple]]:
        """解析批量 LLM 响应为 {index: [KGTriple]} 映射"""
        result_map: dict[int, list[KGTriple]] = {}

        if data is None:
            return result_map

        # 支持数组或 { "results": [...] } 格式
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict) and "results" in data:
            entries = data["results"]
        else:
            return result_map

        if not isinstance(entries, list):
            return result_map

        for entry_idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue

            # 确定该结果对应哪条消息
            msg_idx = entry.get("message_index", entry_idx)
            if not isinstance(msg_idx, int) or msg_idx < 0 or msg_idx >= len(items):
                msg_idx = entry_idx
            if msg_idx >= len(items):
                continue

            raw_triples = entry.get("triples", [])
            if not isinstance(raw_triples, list):
                continue

            text = items[msg_idx].get("text", "")
            parsed: list[KGTriple] = []

            for item_data in raw_triples[:5]:
                triple = self._parse_single_triple(item_data, text)
                if triple:
                    parsed.append(triple)

            if parsed:
                orig_index = items[msg_idx].get("index", msg_idx)
                result_map[orig_index] = parsed

        return result_map

    @staticmethod
    def _parse_single_triple(
        item_data: dict[str, Any], source_text: str
    ) -> KGTriple | None:
        """从单个 JSON dict 解析出一个 KGTriple"""
        if not isinstance(item_data, dict):
            return None

        subject = item_data.get("subject", "").strip()
        obj = item_data.get("object", "").strip()
        predicate = item_data.get("predicate", "").strip()
        if not subject or not obj or not predicate:
            return None

        rt_str = item_data.get("relation_type", "related_to")
        try:
            relation_type = KGRelationType(rt_str)
        except ValueError:
            relation_type = KGRelationType.RELATED_TO

        st_str = item_data.get("subject_type", "unknown")
        ot_str = item_data.get("object_type", "unknown")
        try:
            subject_type = KGNodeType(st_str)
        except ValueError:
            subject_type = _guess_node_type(subject)
        try:
            object_type = KGNodeType(ot_str)
        except ValueError:
            object_type = _guess_node_type(obj)

        return KGTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
            subject_type=subject_type,
            object_type=object_type,
            relation_type=relation_type,
            confidence=float(item_data.get("confidence", 0.6)),
            source_text=source_text,
        )
