"""
主动回复白名单管理

管理群聊的静态白名单和动态白名单。
从 ProactiveReplyManager 中拆分，减少文件行数。
"""
from typing import Dict, List, Optional, Set

from iris_memory.utils.logger import get_logger

logger = get_logger("proactive_whitelist")


class ProactiveWhitelist:
    """群聊白名单管理
    
    支持两种模式：
    - 非白名单模式：使用静态白名单（空列表表示允许所有群聊）
    - 白名单模式：仅允许动态白名单中的群聊
    """
    
    def __init__(
        self,
        group_whitelist: Optional[List[str]] = None,
        group_whitelist_mode: bool = False,
        dynamic_whitelist: Optional[List[str]] = None,
    ):
        self.group_whitelist: List[str] = group_whitelist or []
        self.group_whitelist_mode: bool = group_whitelist_mode
        self._dynamic_whitelist: Set[str] = set(dynamic_whitelist or [])
    
    def is_group_allowed(self, group_id: str) -> bool:
        """检查群聊是否允许主动回复
        
        判断逻辑：
        1. 白名单模式：仅允许动态白名单中的群聊
        2. 非白名单模式：检查静态白名单（空列表表示允许所有）
        """
        group_id_str = str(group_id)
        
        if self.group_whitelist_mode:
            return group_id_str in self._dynamic_whitelist
        
        if self.group_whitelist:
            return group_id_str in self.group_whitelist
        return True
    
    def add_group(self, group_id: str) -> bool:
        """将群聊加入动态白名单
        
        Returns:
            是否成功添加（已存在则返回 False）
        """
        group_id_str = str(group_id)
        if group_id_str in self._dynamic_whitelist:
            return False
        self._dynamic_whitelist.add(group_id_str)
        logger.debug(f"Group {group_id} added to proactive reply whitelist")
        return True
    
    def remove_group(self, group_id: str) -> bool:
        """将群聊从动态白名单移除
        
        Returns:
            是否成功移除（不存在则返回 False）
        """
        group_id_str = str(group_id)
        if group_id_str not in self._dynamic_whitelist:
            return False
        self._dynamic_whitelist.discard(group_id_str)
        logger.debug(f"Group {group_id} removed from proactive reply whitelist")
        return True
    
    def is_group_in_whitelist(self, group_id: str) -> bool:
        """检查群聊是否在动态白名单中"""
        return str(group_id) in self._dynamic_whitelist
    
    def get_whitelist(self) -> List[str]:
        """获取动态白名单列表"""
        return sorted(self._dynamic_whitelist)
    
    def serialize(self) -> List[str]:
        """序列化动态白名单（用于 KV 存储）"""
        return sorted(self._dynamic_whitelist)
    
    def deserialize(self, data: list) -> None:
        """反序列化动态白名单（从 KV 存储加载）"""
        if isinstance(data, list):
            self._dynamic_whitelist = set(str(g) for g in data)
            logger.debug(
                f"Loaded {len(self._dynamic_whitelist)} groups to proactive reply whitelist"
            )
