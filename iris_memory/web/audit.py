"""审计日志"""

from __future__ import annotations

from datetime import datetime

from iris_memory.utils.logger import get_logger

_logger = get_logger("web_audit")


def audit_log(action: str, detail: str = "") -> None:
    """记录审计日志"""
    _logger.info(
        f"[AUDIT] action={action} detail={detail} time={datetime.now().isoformat()}"
    )
