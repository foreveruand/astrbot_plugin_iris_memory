"""Web 模块通用工具函数"""

from __future__ import annotations


def safe_int(
    value: str | None,
    default: int,
    min_val: int = 1,
    max_val: int = 10000,
) -> int:
    """安全解析整数参数，限制范围到 [min_val, max_val]"""
    try:
        n = int(value) if value else default
    except (ValueError, TypeError):
        n = default
    return max(min_val, min(n, max_val))
