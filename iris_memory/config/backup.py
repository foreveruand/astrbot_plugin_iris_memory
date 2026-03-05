"""
配置灾备 — 写入前自动备份上一版本

在配置持久化写入前自动备份当前版本，保留最近 N 个版本，
支持回滚到最近的可用版本。
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# 最大保留备份数
MAX_BACKUPS = 5
BACKUP_SUFFIX = ".bak"


class ConfigBackup:
    """配置文件备份管理器"""

    def __init__(self, config_path: Path, *, max_backups: int = MAX_BACKUPS):
        self._config_path = config_path
        self._max_backups = max_backups
        self._backup_dir = config_path.parent / "config_backups"

    def backup_before_write(self) -> Optional[Path]:
        """在写入新配置前备份当前版本

        Returns:
            备份文件路径，或 None（如果源文件不存在）
        """
        if not self._config_path.exists():
            return None

        self._backup_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        backup_name = f"{self._config_path.stem}.{ts}{BACKUP_SUFFIX}"
        backup_path = self._backup_dir / backup_name

        try:
            shutil.copy2(self._config_path, backup_path)
            self._prune_old_backups()
            logger.debug("配置已备份: %s", backup_path)
            return backup_path
        except OSError:
            logger.exception("配置备份失败")
            return None

    def restore_latest(self) -> Optional[Dict[str, Any]]:
        """恢复最近的有效备份

        Returns:
            恢复的配置字典，或 None
        """
        backups = self._sorted_backups()
        for bp in backups:
            try:
                data = json.loads(bp.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    logger.info("从备份恢复配置: %s", bp)
                    return data
            except (json.JSONDecodeError, OSError):
                logger.warning("备份文件损坏，跳过: %s", bp)
                continue
        return None

    def list_backups(self) -> list:
        """列出所有备份文件（按时间倒序）"""
        return [
            {"path": str(bp), "size": bp.stat().st_size, "mtime": bp.stat().st_mtime}
            for bp in self._sorted_backups()
        ]

    # ── 内部 ──

    def _sorted_backups(self) -> list:
        """返回按修改时间倒序排列的备份文件列表"""
        if not self._backup_dir.exists():
            return []
        backups = list(self._backup_dir.glob(f"*{BACKUP_SUFFIX}"))
        backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return backups

    def _prune_old_backups(self) -> None:
        """清理超出数量限制的旧备份"""
        backups = self._sorted_backups()
        for old in backups[self._max_backups:]:
            try:
                old.unlink()
                logger.debug("清理旧备份: %s", old)
            except OSError:
                pass
