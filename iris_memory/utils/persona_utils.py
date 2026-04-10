"""
人格 ID 提取工具

从 AstrBot 事件中安全提取 persona_id。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent


async def get_event_persona_id(
    event: AstrMessageEvent, context: Any = None
) -> str | None:
    """从事件中提取 persona_id。

    优先从 AstrBot conversation_manager 中读取当前会话绑定的人格 ID（authoritative），
    降级到 event.persona 属性兼容旧版本。

    Args:
        event: AstrBot 消息事件
        context: AstrBot Star Context（可选）。提供时通过 conversation_manager 查询。

    Returns:
        persona_id 字符串，或 None（表示未指定人格，交由调用方使用默认值）
    """
    if context is not None:
        try:
            umo = event.unified_msg_origin
            conv_mgr = context.conversation_manager
            curr_cid = await conv_mgr.get_curr_conversation_id(umo)
            if curr_cid:
                conversation = await conv_mgr.get_conversation(umo, curr_cid)
                if conversation and conversation.persona_id:
                    return conversation.persona_id
        except Exception:
            pass

    # Fallback: try event.persona (compatibility with future AstrBot versions)
    persona = getattr(event, "persona", None)
    if persona is None:
        return None
    persona_id = getattr(persona, "id", None)
    if persona_id and isinstance(persona_id, str) and persona_id.strip():
        return persona_id.strip()
    return None
