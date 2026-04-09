"""Web 用户画像与情感状态服务"""

from __future__ import annotations

from typing import Any

from iris_memory.utils.logger import get_logger

logger = get_logger("web.persona_svc")


class PersonaWebService:
    """Web 端用户画像与情感状态服务"""

    def __init__(self, persona_repo: Any, emotion_repo: Any) -> None:
        self._persona_repo = persona_repo
        self._emotion_repo = emotion_repo

    async def list_personas(
        self,
        query: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        if query:
            return await self._persona_repo.search(
                query=query, page=page, page_size=page_size
            )
        return await self._persona_repo.list_all(page=page, page_size=page_size)

    async def get_persona_detail(self, user_id: str) -> dict[str, Any] | None:
        return await self._persona_repo.get_by_user_id(user_id)

    async def get_emotion_state(
        self,
        user_id: str | None = None,
        group_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not user_id:
            return None
        return await self._emotion_repo.get_by_user_id(user_id)

    async def delete_persona(self, user_id: str) -> dict[str, Any]:
        success, message = await self._persona_repo.delete_by_user_id(user_id)
        return {"success": success, "message": message}

    async def clear_all_personas(self) -> dict[str, Any]:
        success, message, count = await self._persona_repo.clear_all()
        return {"success": success, "message": message, "count": count}
