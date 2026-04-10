"""
人格 ID 提取工具

从 AstrBot 事件中安全提取 persona_id。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("iris_memory.persona")

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent


async def get_event_persona_id(
    event: AstrMessageEvent, context: Any = None
) -> str | None:
    """从事件中提取当前生效的 persona_id。

    使用 AstrBot PersonaManager.resolve_selected_persona() 作为权威来源，
    与 AstrBot 主流程保持完全一致。

    Args:
        event: AstrBot 消息事件
        context: AstrBot Star Context（可选）。提供时通过 persona_manager 解析。

    Returns:
        persona_id 字符串，或 None（表示未指定人格，交由调用方使用默认值）
    """
    if context is None:
        logger.debug("[persona] context is None, returning None")
        return None

    try:
        umo = event.unified_msg_origin

        # Step 1: get the conversation's explicit persona_id (may be None)
        conv_persona_id = None
        try:
            conv_mgr = context.conversation_manager
            curr_cid = await conv_mgr.get_curr_conversation_id(umo)
            if curr_cid:
                conversation = await conv_mgr.get_conversation(umo, curr_cid)
                if conversation:
                    conv_persona_id = conversation.persona_id
            logger.debug(
                "[persona] umo=%s cid=%s conv_persona_id=%s",
                umo,
                curr_cid,
                conv_persona_id,
            )
        except Exception as e:
            logger.debug("[persona] conversation_manager lookup failed: %s", e)

        # Step 2: use resolve_selected_persona for the authoritative result
        # (same logic AstrBot's main agent uses)
        try:
            acm = context.astrbot_config_mgr
            cfg = acm.get_conf(umo) if acm is not None else {}
            provider_settings = cfg.get("provider_settings", {}) if cfg else {}
            logger.debug(
                "[persona] provider_settings.default_personality=%s",
                provider_settings.get("default_personality"),
            )

            (resolved_persona_id, persona_obj, _, _) = (
                await context.persona_manager.resolve_selected_persona(
                    umo=umo,
                    conversation_persona_id=conv_persona_id,
                    platform_name=event.get_platform_name(),
                    provider_settings=provider_settings,
                )
            )
            logger.debug(
                "[persona] resolved_persona_id=%s persona_name=%s",
                resolved_persona_id,
                persona_obj.get("name") if persona_obj else None,
            )

            # Filter out special internal sentinel values
            if resolved_persona_id and resolved_persona_id not in (
                "[%None]",
                "_chatui_default_",
            ):
                return resolved_persona_id

        except Exception as e:
            logger.debug("[persona] resolve_selected_persona failed: %s", e)

    except Exception as e:
        logger.debug("[persona] get_event_persona_id outer error: %s", e)

    return None

