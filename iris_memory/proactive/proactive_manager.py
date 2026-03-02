"""
主动回复管理器
协调检测、事件注入整个流程

当前架构：
不再自行调用 LLM 生成回复和直接发送，
而是构造合成事件注入 AstrBot 事件队列，
让主动回复经过完整的 Pipeline 处理流程：
  人格注入 → 插件 Hook（记忆检索等）→ LLM 生成 → 结果装饰 → 发送

组合模块：
- ProactiveConstraints: 冷却、每日限制、连续回复限制、启动冷却
- ProactiveWhitelist: 群聊白名单管理
- SmartBoostManager: 智能增强窗口与决策调整
"""
import asyncio
import time
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass
from typing import Protocol

from iris_memory.utils.logger import get_logger
from iris_memory.utils.command_utils import SessionKeyBuilder
from iris_memory.proactive.proactive_reply_detector import (
    ProactiveReplyDetector, ProactiveReplyDecision, ReplyUrgency
)
from iris_memory.proactive.proactive_event import ProactiveMessageEvent
from iris_memory.proactive.proactive_constraints import ProactiveConstraints
from iris_memory.proactive.proactive_whitelist import ProactiveWhitelist
from iris_memory.proactive.proactive_smart_boost import SmartBoostManager

if TYPE_CHECKING:
    from iris_memory.core.config_manager import ConfigManager
    from iris_memory.proactive.llm_proactive_reply_detector import LLMReplyDecision


class ReplyDecisionProtocol(Protocol):
    """主动回复决策协议

    定义 ProactiveReplyDecision 和 LLMReplyDecision 的共同接口。
    使用 Protocol 实现结构化子类型（鸭子类型），避免修改现有继承关系。
    """
    should_reply: bool
    urgency: ReplyUrgency
    reason: str
    suggested_delay: int

logger = get_logger("proactive_manager")


@dataclass
class ProactiveReplyTask:
    """主动回复任务"""
    messages: List[str]
    user_id: str
    group_id: Optional[str]
    decision: ProactiveReplyDecision
    context: Dict[str, Any]
    umo: str = ""
    persona_id: str = "default"


class ProactiveReplyManager:
    """主动回复管理器

    通过组合模式委托限制检查、白名单管理和智能增强逻辑。
    核心职责：检测 → 约束检查 → 入队 → 任务处理 → 事件注入。
    """

    def __init__(
        self,
        astrbot_context=None,
        reply_detector: Optional[ProactiveReplyDetector] = None,
        event_queue: Optional[asyncio.Queue] = None,
        config: Optional[Dict] = None,
        config_manager: Optional['ConfigManager'] = None
    ):
        self.config = config or {}
        self.reply_detector = reply_detector
        self.astrbot_context = astrbot_context
        self.event_queue = event_queue
        self._config_manager = config_manager

        # 配置
        self.enabled = self.config.get("enable_proactive_reply", True)

        # 组合模块
        self._constraints = ProactiveConstraints(
            config_manager=config_manager,
            default_cooldown=self.config.get("reply_cooldown", 60),
            default_max_daily=self.config.get("max_daily_replies", 20),
        )
        self._whitelist = ProactiveWhitelist(
            group_whitelist=self._parse_list_config("group_whitelist"),
            group_whitelist_mode=self.config.get("group_whitelist_mode", False),
            dynamic_whitelist=self._parse_list_config("dynamic_whitelist"),
        )
        self.stats = {
            "replies_sent": 0,
            "replies_skipped": 0,
            "replies_failed": 0,
            "replies_consecutive_limited": 0,
            "tasks_cleared_on_bot_reply": 0,
            "smart_boost_activations": 0,
            "smart_boost_delay_reductions": 0,
        }

        self._smart_boost = SmartBoostManager(
            config=self.config,
            config_manager=config_manager,
            stats=self.stats,
        )

        # 任务队列
        self.pending_tasks: asyncio.Queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        self.is_running = False

        # 已排队/处理中的会话（防止同一会话重复入队）
        self._queued_sessions: set = set()
        self._processing_sessions: set = set()

        # 群冷却检查回调：(group_id: str) -> bool
        self._cooldown_checker: Optional[Any] = None

    def _parse_list_config(self, key: str) -> List[str]:
        """统一解析列表类型配置

        Args:
            key: 配置键名

        Returns:
            字符串列表，自动处理 str/list/其他类型
        """
        value = self.config.get(key, [])
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, list):
            return [str(v) for v in value if v]
        return []

    # ========== 兼容性代理属性 ==========
    # 以下属性保持向后兼容，委托给组合模块

    @property
    def last_reply_time(self):
        return self._constraints.last_reply_time

    @property
    def daily_reply_count(self):
        return self._constraints.daily_reply_count

    @property
    def _last_user_message_time(self):
        return self._smart_boost._last_user_message_time

    @property
    def _recent_replies(self):
        return self._constraints._recent_replies

    @property
    def MAX_CONSECUTIVE_REPLIES(self) -> int:
        return self._constraints.MAX_CONSECUTIVE_REPLIES

    @MAX_CONSECUTIVE_REPLIES.setter
    def MAX_CONSECUTIVE_REPLIES(self, value: int) -> None:
        self._constraints.MAX_CONSECUTIVE_REPLIES = value

    @property
    def CONSECUTIVE_WINDOW(self) -> int:
        return self._constraints.CONSECUTIVE_WINDOW

    @CONSECUTIVE_WINDOW.setter
    def CONSECUTIVE_WINDOW(self, value: int) -> None:
        self._constraints.CONSECUTIVE_WINDOW = value

    @property
    def _startup_time(self):
        return self._constraints._startup_time

    @_startup_time.setter
    def _startup_time(self, value):
        self._constraints._startup_time = value

    @property
    def STARTUP_COOLDOWN_SECONDS(self) -> int:
        return self._constraints.STARTUP_COOLDOWN_SECONDS

    @STARTUP_COOLDOWN_SECONDS.setter
    def STARTUP_COOLDOWN_SECONDS(self, value: int) -> None:
        self._constraints.STARTUP_COOLDOWN_SECONDS = value

    @property
    def _default_cooldown(self):
        return self._constraints._default_cooldown

    @_default_cooldown.setter
    def _default_cooldown(self, value):
        self._constraints._default_cooldown = value

    @property
    def _default_max_daily(self):
        return self._constraints._default_max_daily

    @_default_max_daily.setter
    def _default_max_daily(self, value):
        self._constraints._default_max_daily = value

    @property
    def _last_reset_date(self):
        return self._constraints._last_reset_date

    @_last_reset_date.setter
    def _last_reset_date(self, value):
        self._constraints._last_reset_date = value

    @property
    def group_whitelist(self):
        return self._whitelist.group_whitelist

    @group_whitelist.setter
    def group_whitelist(self, value):
        self._whitelist.group_whitelist = value

    @property
    def group_whitelist_mode(self):
        return self._whitelist.group_whitelist_mode

    @group_whitelist_mode.setter
    def group_whitelist_mode(self, value):
        self._whitelist.group_whitelist_mode = value

    @property
    def _dynamic_whitelist(self):
        return self._whitelist._dynamic_whitelist

    @_dynamic_whitelist.setter
    def _dynamic_whitelist(self, value):
        self._whitelist._dynamic_whitelist = value

    @property
    def _smart_boost_enabled(self) -> bool:
        return self._smart_boost.enabled

    @property
    def _smart_boost_window(self) -> int:
        return self._smart_boost.window_seconds

    @property
    def _smart_boost_multiplier(self) -> float:
        return self._smart_boost.multiplier

    @property
    def _smart_boost_threshold(self) -> float:
        return self._smart_boost.threshold

    # ========== 生命周期 ==========

    async def initialize(self):
        """初始化"""
        if not self.enabled:
            logger.debug("Proactive reply is disabled")
            return

        if not self.event_queue:
            if self.astrbot_context and hasattr(self.astrbot_context, '_event_queue'):
                self.event_queue = self.astrbot_context._event_queue
            if not self.event_queue:
                logger.debug("Event queue not available, proactive reply disabled")
                self.enabled = False
                return

        self.is_running = True
        self._constraints._startup_time = time.time()
        self.processing_task = asyncio.create_task(self._process_loop())

        logger.debug(
            f"Proactive reply manager initialized (event queue mode, "
            f"startup cooldown: {self._constraints.STARTUP_COOLDOWN_SECONDS}s)"
        )

    async def stop(self):
        """停止（热更新友好）"""
        logger.debug("[Hot-Reload] Stopping ProactiveReplyManager...")
        self.is_running = False

        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"[Hot-Reload] Error cancelling processing task: {e}")
            self.processing_task = None

        while not self.pending_tasks.empty():
            try:
                task = self.pending_tasks.get_nowait()
                await self._process_task(task, skip_delay=True)
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.error(f"Error processing pending task during shutdown: {e}")

        logger.debug("[Hot-Reload] ProactiveReplyManager stopped")

    # ========== 任务管理 ==========

    def clear_pending_tasks_for_session(
        self,
        user_id: str,
        group_id: Optional[str] = None,
    ) -> int:
        """清除指定会话的待处理任务"""
        session_key = SessionKeyBuilder.build(user_id, group_id)
        cleared_count = 0
        remaining_tasks = []

        while not self.pending_tasks.empty():
            try:
                task = self.pending_tasks.get_nowait()
                task_session_key = SessionKeyBuilder.build(task.user_id, task.group_id)
                if task_session_key != session_key:
                    remaining_tasks.append(task)
                else:
                    cleared_count += 1
            except asyncio.QueueEmpty:
                break

        for task in remaining_tasks:
            self.pending_tasks.put_nowait(task)

        if cleared_count > 0:
            self.stats["tasks_cleared_on_bot_reply"] += cleared_count
            logger.debug(
                f"Cleared {cleared_count} pending proactive tasks for {session_key} "
                f"(Bot already replied)"
            )

        return cleared_count

    def clear_all_pending_tasks(self) -> int:
        """清除所有待处理任务"""
        cleared_count = self.pending_tasks.qsize()
        while not self.pending_tasks.empty():
            try:
                self.pending_tasks.get_nowait()
            except asyncio.QueueEmpty:
                break
        return cleared_count

    # ========== 核心流程 ==========

    async def handle_batch(
        self,
        messages: List[str],
        user_id: str,
        group_id: Optional[str] = None,
        context: Optional[Dict] = None,
        umo: str = "",
        persona_id: str = "default"
    ):
        """处理批量消息，判断是否需要主动回复"""
        if not self.enabled or not messages:
            return

        if self._constraints.is_in_startup_cooldown():
            logger.debug("Proactive reply in startup cooldown, skipping")
            return

        if group_id and self._cooldown_checker and self._cooldown_checker(group_id):
            logger.debug(f"Group {group_id} is in cooldown, skipping proactive reply")
            return

        session_key = SessionKeyBuilder.build(user_id, group_id)
        self._smart_boost.record_user_message(user_id, group_id)

        if self._constraints.is_daily_limit_reached(user_id, group_id):
            logger.debug(f"Daily proactive reply limit reached for {user_id}")
            return

        if self._constraints.is_consecutive_limit_reached(session_key):
            logger.debug(
                f"Consecutive reply limit reached for {session_key} "
                f"({self._constraints.MAX_CONSECUTIVE_REPLIES} "
                f"in {self._constraints.CONSECUTIVE_WINDOW}s)"
            )
            self.stats["replies_consecutive_limited"] += 1
            return

        if group_id and not self._whitelist.is_group_allowed(group_id):
            logger.debug(f"Group {group_id} not allowed for proactive reply, skipping")
            return

        if not self.reply_detector:
            return

        try:
            decision = await self.reply_detector.analyze(
                messages=messages,
                user_id=user_id,
                group_id=group_id,
                context=context,
            )

            decision = self._smart_boost.apply(decision, user_id, group_id)

            if decision.should_reply:
                if self._constraints.is_in_cooldown(
                    session_key, group_id, urgency=decision.urgency.value
                ):
                    logger.debug(
                        f"Proactive reply in cooldown for {session_key} "
                        f"(urgency={decision.urgency.value})"
                    )
                    return

                if session_key in self._queued_sessions:
                    logger.debug(f"Task already queued for {session_key}, skipping")
                    self.stats["replies_skipped"] += 1
                    return

                task = ProactiveReplyTask(
                    messages=messages,
                    user_id=user_id,
                    group_id=group_id,
                    decision=decision,
                    context=context or {},
                    umo=umo,
                    persona_id=persona_id,
                )

                await self.pending_tasks.put(task)
                self._queued_sessions.add(session_key)

                logger.debug(
                    f"Proactive reply queued for {session_key}, "
                    f"urgency: {decision.urgency.value}"
                )
            else:
                self.stats["replies_skipped"] += 1

        except Exception as e:
            logger.error(f"Error in proactive reply detection: {e}")

    async def _process_loop(self):
        """处理循环"""
        while self.is_running:
            try:
                task = await asyncio.wait_for(
                    self.pending_tasks.get(),
                    timeout=1.0,
                )
                await self._process_task(task)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except RuntimeError as e:
                logger.error(f"Runtime error in process loop: {e}")
                await asyncio.sleep(1)
            except Exception as e:
                logger.exception(f"Unexpected error in process loop: {e}")
                await asyncio.sleep(5)

    async def _process_task(self, task: ProactiveReplyTask, skip_delay: bool = False):
        """处理回复任务：构造合成事件并注入事件队列"""
        session_key = SessionKeyBuilder.build(task.user_id, task.group_id)

        try:
            delay = task.decision.suggested_delay
            if delay > 0 and not skip_delay:
                await asyncio.sleep(delay)

            if session_key in self._processing_sessions:
                logger.debug(
                    f"Proactive reply cancelled, another task is processing for {session_key}"
                )
                self.stats["replies_skipped"] += 1
                return

            self._processing_sessions.add(session_key)

            if not self.event_queue or not self.astrbot_context:
                logger.debug("Event queue or context not available, skip proactive reply")
                self.stats["replies_failed"] += 1
                return

            trigger_prompt = self._build_trigger_prompt(task)
            if not trigger_prompt:
                logger.warning(f"Failed to build trigger prompt for {task.user_id}")
                self.stats["replies_failed"] += 1
                return

            proactive_event = self._build_proactive_event(task, trigger_prompt)

            try:
                self.event_queue.put_nowait(proactive_event)
            except asyncio.QueueFull:
                logger.warning(
                    f"Event queue full, proactive reply for {task.user_id} dropped"
                )
                self.stats["replies_failed"] += 1
                return

            self.stats["replies_sent"] += 1
            self._constraints.last_reply_time[session_key] = (
                asyncio.get_running_loop().time()
            )
            self._constraints.record_reply_time(session_key)
            self._constraints.increment_daily_count(task.user_id, task.group_id)

            logger.debug(
                f"Proactive reply event dispatched for {task.user_id}, "
                f"urgency: {task.decision.urgency.value}, "
                f"reason: {task.decision.reason}"
            )

        except Exception as e:
            logger.error(f"Error processing proactive reply task: {e}")
            self.stats["replies_failed"] += 1
        finally:
            self._processing_sessions.discard(session_key)
            self._queued_sessions.discard(session_key)

    # ========== 事件构造 ==========

    def _build_proactive_event(
        self,
        task: ProactiveReplyTask,
        trigger_prompt: str,
    ) -> ProactiveMessageEvent:
        """构造主动回复合成事件"""
        sender_name = task.context.get("sender_name", "")

        recent_messages = [
            {"sender_name": sender_name or task.user_id, "content": msg[:200]}
            for msg in task.messages[-5:]
        ]

        reply_context = task.decision.reply_context or {}
        emotion_data = reply_context.get("emotion", {})
        emotion_summary = ""
        if emotion_data:
            primary = emotion_data.get("primary", "")
            intensity = emotion_data.get("intensity", 0)
            if primary:
                emotion_summary = f"{primary}（强度 {intensity:.1f}）"

        proactive_context = {
            "reason": task.decision.reason,
            "urgency": task.decision.urgency.value,
            "reply_context": reply_context,
            "message_count": len(task.messages),
            "user_id": task.user_id,
            "group_id": task.group_id,
            "recent_messages": recent_messages,
            "emotion_summary": emotion_summary,
            "target_user": sender_name or task.user_id,
        }

        return ProactiveMessageEvent(
            context=self.astrbot_context,
            umo=task.umo,
            trigger_prompt=trigger_prompt,
            user_id=task.user_id,
            sender_name=sender_name,
            group_id=task.group_id,
            proactive_context=proactive_context,
            persona_id=task.persona_id,
        )

    def _build_trigger_prompt(self, task: ProactiveReplyTask) -> str:
        """构建触发提示词"""
        reply_context = task.decision.reply_context or {}
        reason = reply_context.get("reason", task.decision.reason)

        recent_messages = task.messages[-5:] if task.messages else []
        messages_summary = "\n".join(f"- {msg[:100]}" for msg in recent_messages)

        emotion_info = ""
        emotion_data = reply_context.get("emotion", {})
        if emotion_data:
            primary = emotion_data.get("primary", "")
            intensity = emotion_data.get("intensity", 0)
            if primary:
                emotion_info = f"\n用户当前情绪：{primary}（强度 {intensity:.1f}）"

        return (
            f"你现在要主动在群聊进行发言。\n"
            f"发言原因：{reason}\n"
            f"用户最近的消息：\n{messages_summary}"
            f"{emotion_info}\n\n"
            f"请根据你的人格和与群聊用户的记忆，生成一条自然、简短的主动消息。"
            f"像朋友一样自然地接话，不要说明你是在主动回复。\n"
            f"重要提示：\n"
            f"- 不要重复提及用户刚才已经说过的话题或事件\n"
            f"- 如果是群聊，注意不要过度介入，保持适度存在感\n"
            f"- 回复内容保持自己的人格\n"
            f"- 避免机械式回应，要有个性化的互动"
        )

    # ========== 委托方法（保持向后兼容） ==========

    def _record_user_message(self, user_id: str, group_id: Optional[str] = None) -> None:
        """记录用户发言时间"""
        self._smart_boost.record_user_message(user_id, group_id)

    def is_in_boost_window(self, user_id: str, group_id: Optional[str] = None) -> bool:
        """检查是否在智能增强窗口内"""
        return self._smart_boost.is_in_boost_window(user_id, group_id)

    def get_boost_multiplier(self, user_id: str, group_id: Optional[str] = None) -> float:
        """获取当前智能增强乘数"""
        return self._smart_boost.get_boost_multiplier(user_id, group_id)

    def _apply_smart_boost(self, decision, user_id, group_id=None):
        """应用智能增强"""
        return self._smart_boost.apply(decision, user_id, group_id)

    def _is_in_cooldown(self, session_key, group_id=None, urgency=None):
        """检查冷却"""
        return self._constraints.is_in_cooldown(session_key, group_id, urgency)

    def _check_daily_reset(self):
        """检查每日重置"""
        self._constraints.check_daily_reset()

    def _is_daily_limit_reached(self, user_id, group_id=None):
        """检查每日限制"""
        return self._constraints.is_daily_limit_reached(user_id, group_id)

    def _is_consecutive_limit_reached(self, session_key):
        """检查连续回复限制"""
        return self._constraints.is_consecutive_limit_reached(session_key)

    def _record_reply_time(self, session_key):
        """记录回复时间"""
        self._constraints.record_reply_time(session_key)

    def _is_in_startup_cooldown(self):
        """检查启动冷却"""
        return self._constraints.is_in_startup_cooldown()

    def _is_group_allowed(self, group_id):
        """检查群聊是否允许"""
        return self._whitelist.is_group_allowed(group_id)

    def add_group_to_whitelist(self, group_id):
        """加入白名单"""
        return self._whitelist.add_group(group_id)

    def remove_group_from_whitelist(self, group_id):
        """移出白名单"""
        return self._whitelist.remove_group(group_id)

    def is_group_in_whitelist(self, group_id):
        """检查是否在白名单"""
        return self._whitelist.is_group_in_whitelist(group_id)

    def get_whitelist(self):
        """获取白名单"""
        return self._whitelist.get_whitelist()

    def serialize_whitelist(self):
        """序列化白名单"""
        return self._whitelist.serialize()

    def deserialize_whitelist(self, data):
        """反序列化白名单"""
        self._whitelist.deserialize(data)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "pending_tasks": self.pending_tasks.qsize(),
            "last_reply_times": len(self._constraints.last_reply_time),
            "daily_counts": self._constraints.daily_reply_count.copy(),
        }

    def reset_daily_counts(self):
        """重置每日计数"""
        self._constraints.reset_daily_counts()

    def _get_cooldown_seconds(self, group_id=None):
        """获取冷却秒数"""
        return self._constraints.get_cooldown_seconds(group_id)

    def _get_max_daily_replies(self, group_id=None):
        """获取每日最大回复数"""
        return self._constraints.get_max_daily_replies(group_id)
