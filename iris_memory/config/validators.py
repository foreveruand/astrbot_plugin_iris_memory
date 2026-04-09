"""
配置校验器 — 运行时完整性与有效性检查

基于 Schema 定义执行类型检查、范围校验和枚举校验。
类似 Zod 的 ``parse()``：传入原始值，返回校验后的安全值或抛出异常。
"""

from __future__ import annotations

from typing import Any

from iris_memory.config.schema import SCHEMA, ConfigField


class ConfigValidationError(Exception):
    """配置校验失败"""

    def __init__(self, errors: list[tuple[str, str]]):
        self.errors = errors
        msg = "; ".join(f"[{k}] {m}" for k, m in errors)
        super().__init__(f"配置校验失败: {msg}")


def validate_field(field: ConfigField, value: Any) -> Any:
    """校验并规范化单个配置值

    Returns:
        校验通过后的值（可能经过类型转换）

    Raises:
        ValueError: 校验失败
    """
    if value is None:
        return field.default

    # ── 类型转换 ──
    if field.value_type is not object:
        value = _coerce_type(field.key, value, field.value_type)

    # ── 枚举校验 ──
    if field.choices is not None and value not in field.choices:
        raise ValueError(f"值 {value!r} 不在允许范围 {field.choices} 内")

    # ── 范围校验 ──
    if field.min_val is not None and isinstance(value, (int, float)):
        if value < field.min_val:
            raise ValueError(f"值 {value} 小于最小值 {field.min_val}")
    if field.max_val is not None and isinstance(value, (int, float)):
        if value > field.max_val:
            raise ValueError(f"值 {value} 大于最大值 {field.max_val}")

    # ── 自定义校验 ──
    if field.validator is not None:
        value = field.validator(value)

    return value


def validate_dict(data: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    """校验整个配置字典

    对 ``data`` 中每个键按 Schema 校验并注入缺失键的默认值。

    Args:
        data: 待校验的扁平化配置字典
        strict: True 时遇到首个错误即抛异常；False 时收集所有错误

    Returns:
        校验后的完整配置字典

    Raises:
        ConfigValidationError: 校验失败（仅 strict=True 或收集到错误时）
    """
    result: dict[str, Any] = {}
    errors: list[tuple[str, str]] = []

    for key, field in SCHEMA.items():
        raw = data.get(key)
        try:
            result[key] = validate_field(field, raw)
        except (ValueError, TypeError) as exc:
            if strict:
                raise ConfigValidationError([(key, str(exc))]) from exc
            errors.append((key, str(exc)))
            result[key] = field.default  # 降级到默认值

    if errors:
        raise ConfigValidationError(errors)

    return result


def inject_defaults(data: dict[str, Any]) -> dict[str, Any]:
    """为缺失的键注入默认值（不做校验）"""
    result = dict(data)
    for key, field in SCHEMA.items():
        if key not in result:
            result[key] = field.default
    return result


# ─── 内部 ────────────────────────────────────────────


def _coerce_type(key: str, value: Any, target: type) -> Any:
    """尝试将 value 转换为目标类型"""
    if isinstance(value, target):
        return value
    try:
        if target is bool:
            # 特殊处理布尔：避免 bool("false") == True
            if isinstance(value, str):
                if value.lower() in ("true", "1", "yes", "on"):
                    return True
                if value.lower() in ("false", "0", "no", "off"):
                    return False
                raise ValueError(f"无法将 {value!r} 转换为布尔值")
            return bool(value)
        return target(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"类型转换失败: 期望 {target.__name__}，得到 {type(value).__name__} ({value!r})"
        ) from exc
