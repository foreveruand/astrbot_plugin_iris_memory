"""
统一 LLM 调用工具

将散布在 llm_processor / llm_extractor / llm_enhanced_base 中的
``llm_generate → text_chat`` fallback 模式和 JSON 解析逻辑
集中到一处，避免散弹修改。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from iris_memory.core.constants import SOURCE_ALIASES
from iris_memory.core.provider_utils import (
    extract_provider_id,
    get_default_provider,
    get_provider_by_id,
    normalize_provider_id,
)
from iris_memory.utils.llm_rate_controller import get_rate_controller
from iris_memory.utils.logger import get_logger

logger = get_logger("llm_helper")


# ── Provider / Context 协议 ────────────────────────────────


@runtime_checkable
class LLMProvider(Protocol):
    """LLM 提供者协议"""

    async def text_chat(
        self,
        *,
        prompt: str,
        context: list[Any],
        image_urls: list[str] | None = None,
    ) -> Any: ...


@runtime_checkable
class AstrBotContext(Protocol):
    """AstrBot 上下文协议（仅 LLM 相关部分）"""

    async def llm_generate(self, *, chat_provider_id: str, prompt: str) -> Any: ...


# 延迟加载 token 估算（避免循环导入）
_token_estimator = None


def _estimate_tokens(text: str) -> int:
    """使用 token_manager 的精确估算（tiktoken 优先，加权启发式兜底）"""
    global _token_estimator
    if _token_estimator is None:
        try:
            from iris_memory.utils.token_manager import TokenBudget

            _token_estimator = TokenBudget()
        except Exception:
            _token_estimator = "fallback"
    if _token_estimator == "fallback":
        # 简单加权估算：中文 ~1.5 char/token, 英文 ~4 char/token
        chinese = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        other = len(text) - chinese
        return int(chinese / 1.5 + other / 4) or 1
    return _token_estimator.estimate_tokens(text)


# ── 数据类型 ────────────────────────────────────────────


@dataclass
class LLMCallResult:
    """LLM 调用结果"""

    success: bool = False
    content: str = ""
    parsed_json: dict[str, Any] | None = None
    tokens_used: int = 0
    error: str = ""


# ── JSON 解析 ────────────────────────────────────────────


def parse_llm_json(response: str | None) -> dict[str, Any] | None:
    """从 LLM 响应文本中提取 JSON 字典。

    三级尝试：
    1. 直接 ``json.loads``
    2. 从 markdown 代码块 (````json ... ````) 提取
    3. 匹配第一个 ``{...}`` 子串

    Returns:
        解析成功返回 dict，否则 None。
    """
    if not response:
        return None

    # 1) 直接尝试
    try:
        result = json.loads(response)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # 2) code-fenced JSON
    try:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if m:
            result = json.loads(m.group(1))
            if isinstance(result, dict):
                return result
    except (json.JSONDecodeError, TypeError):
        pass

    # 3) 裸 {...}（非贪婪）
    try:
        m = re.search(r"(\{[\s\S]*?\})", response)
        if m:
            result = json.loads(m.group(1))
            if isinstance(result, dict):
                return result
    except (json.JSONDecodeError, TypeError):
        pass

    return None


# ── Provider 解析 ────────────────────────────────────────


def resolve_llm_provider(
    context: AstrBotContext | None,
    provider_id: str = "",
    *,
    label: str = "LLM",
) -> tuple[LLMProvider | None, str | None]:
    """解析 LLM 提供者。

    Args:
        context: AstrBot 上下文
        provider_id: 期望的 provider ID（"" / "default" 表示使用默认）
        label: 用于日志的标签

    Returns:
        (provider, resolved_id) 或 (None, None)
    """
    if not context:
        return None, None

    pid = normalize_provider_id(provider_id)
    logger.debug(
        f"[{label}] resolve_llm_provider: input={repr(provider_id)}, normalized={repr(pid)}"
    )

    if pid and pid not in ("", "default"):
        try:
            provider, resolved_id = get_provider_by_id(context, pid)
            if provider:
                logger.debug(f"[{label}] Provider resolved by ID: {resolved_id or pid}")
                return provider, resolved_id or pid
            logger.warning(
                f"[{label}] Provider '{pid}' not found, falling back to default"
            )
        except Exception as e:
            logger.warning(f"[{label}] Error resolving provider '{pid}': {e}")
    else:
        logger.debug(
            f"[{label}] pid is empty or 'default', using AstrBot default provider"
        )

    provider, resolved_id = get_default_provider(context)
    if provider:
        resolved_id = resolved_id or extract_provider_id(provider)
        logger.debug(f"[{label}] Using default provider: {resolved_id}")
    return provider, resolved_id


# ── 单次 LLM 调用 ────────────────────────────────────────


async def _call_llm_single(
    context: AstrBotContext | None,
    provider: LLMProvider | None,
    provider_id: str | None,
    prompt: str,
    *,
    parse_json: bool = False,
    image_urls: list[str] | None = None,
    source_module: str = "unknown",
    source_class: str = "unknown",
) -> LLMCallResult:
    start_time = time.time()
    is_multimodal = bool(image_urls)
    result = LLMCallResult(success=False, error="No suitable LLM method found")

    rate_controller = get_rate_controller()

    try:
        if (
            not is_multimodal
            and context
            and hasattr(context, "llm_generate")
            and provider_id
        ):
            try:
                await rate_controller.acquire(provider_id)
                try:
                    logger.debug(
                        f"Calling context.llm_generate with provider_id={repr(provider_id)}, prompt_length={len(prompt)}"
                    )
                    resp = await context.llm_generate(
                        chat_provider_id=provider_id,
                        prompt=prompt,
                    )
                    if resp and hasattr(resp, "completion_text"):
                        text = (resp.completion_text or "").strip()
                        tokens = _estimate_tokens(prompt + text)
                        logger.debug(
                            f"llm_generate success, response_length={len(text)}, tokens={tokens}"
                        )
                        result = LLMCallResult(
                            success=True,
                            content=text,
                            parsed_json=parse_llm_json(text) if parse_json else None,
                            tokens_used=tokens,
                        )
                finally:
                    rate_controller.release(provider_id)
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                logger.warning(
                    f"llm_generate failed: [{error_type}] {error_msg} | "
                    f"provider_id={repr(provider_id)}, prompt_length={len(prompt)}, "
                    f"prompt_preview={repr(prompt[:100])}..."
                )
                result = LLMCallResult(
                    success=False,
                    error=f"[{error_type}] {error_msg}",
                )

        if not result.success and provider and hasattr(provider, "text_chat"):
            resolved_pid = provider_id or extract_provider_id(provider) or "unknown"
            try:
                await rate_controller.acquire(resolved_pid)
                try:
                    kwargs: dict[str, Any] = {"prompt": prompt, "context": []}
                    if is_multimodal:
                        kwargs["image_urls"] = image_urls
                        logger.debug(
                            f"Calling provider.text_chat with multimodal, "
                            f"provider_id={repr(resolved_pid)}, prompt_length={len(prompt)}, "
                            f"image_count={len(image_urls)}"
                        )
                    resp = await provider.text_chat(**kwargs)
                    text = _extract_text(resp)
                    tokens = _estimate_tokens(prompt + text)
                    result = LLMCallResult(
                        success=True,
                        content=text,
                        parsed_json=parse_llm_json(text) if parse_json else None,
                        tokens_used=tokens,
                    )
                finally:
                    rate_controller.release(resolved_pid)
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                logger.warning(
                    f"text_chat failed: [{error_type}] {error_msg} | "
                    f"provider_id={repr(resolved_pid)}, prompt_length={len(prompt)}, "
                    f"prompt_preview={repr(prompt[:100])}..."
                )
                result = LLMCallResult(
                    success=False,
                    error=f"[{error_type}] {error_msg}",
                )
    finally:
        duration_ms = (time.time() - start_time) * 1000
        _record_stats(
            provider_id=provider_id
            or (extract_provider_id(provider) if provider else None),
            result=result,
            duration_ms=duration_ms,
            prompt=prompt,
            is_multimodal=is_multimodal,
            image_count=len(image_urls) if image_urls else 0,
            source_module=source_module,
            source_class=source_class,
        )

    return result


async def call_llm(
    context: AstrBotContext | None,
    provider: LLMProvider | None,
    provider_id: str | None,
    prompt: str,
    *,
    parse_json: bool = False,
    image_urls: list[str] | None = None,
    fallback_provider_ids: list[str] | None = None,
) -> LLMCallResult:
    """统一的 LLM 调用（支持显式 Provider + fallback 轮换）。

    Args:
        context: AstrBot 上下文
        provider: 已解析的 provider 实例
        provider_id: 主要 provider ID
        prompt: 提示词
        parse_json: 是否尝试从响应中解析 JSON
        image_urls: 图片 URL 列表
        fallback_provider_ids: 失败后依次尝试的 fallback provider ID 列表

    Returns:
        LLMCallResult
    """
    source_module, source_class = _infer_caller_source()

    primary_provider = provider
    primary_provider_id = normalize_provider_id(provider_id)

    if primary_provider is None and context:
        primary_provider, resolved_id = resolve_llm_provider(
            context,
            primary_provider_id,
            label="LLM",
        )
        if resolved_id:
            primary_provider_id = resolved_id

    if primary_provider is None and context and not primary_provider_id:
        primary_provider, primary_provider_id = get_default_provider(context)

    primary_result = await _call_llm_single(
        context=context,
        provider=primary_provider,
        provider_id=primary_provider_id,
        prompt=prompt,
        parse_json=parse_json,
        image_urls=image_urls,
        source_module=source_module,
        source_class=source_class,
    )

    if primary_result.success or not fallback_provider_ids:
        return primary_result

    seen_ids: set[str] = set()
    if primary_provider_id:
        seen_ids.add(primary_provider_id)
    elif primary_provider:
        extracted = extract_provider_id(primary_provider)
        if extracted:
            seen_ids.add(extracted)

    total = 1 + len(fallback_provider_ids)
    last_result = primary_result

    for idx, pid in enumerate(fallback_provider_ids):
        normalized_pid = normalize_provider_id(pid)
        if not normalized_pid or normalized_pid in seen_ids:
            continue
        seen_ids.add(normalized_pid)

        fallback_provider, resolved_id = (
            get_provider_by_id(context, normalized_pid) if context else (None, None)
        )
        if not fallback_provider:
            logger.debug(f"Fallback provider '{normalized_pid}' not found, skipping")
            continue

        logger.warning(
            f"Switching to fallback provider: {resolved_id or normalized_pid} "
            f"({idx + 2}/{total})"
        )

        last_result = await _call_llm_single(
            context=context,
            provider=fallback_provider,
            provider_id=resolved_id or normalized_pid,
            prompt=prompt,
            parse_json=parse_json,
            image_urls=image_urls,
            source_module=source_module,
            source_class=source_class,
        )

        if last_result.success:
            logger.info(
                f"LLM call succeeded with fallback provider: {resolved_id or normalized_pid}"
            )
            return last_result

    logger.error(f"All {total} provider(s) failed for LLM call")
    return last_result


def _infer_caller_source() -> tuple[str, str]:
    """从调用栈推断来源（在调用时立即捕获，而非异步任务中）

    Returns:
        (source_alias, source_class)
    """
    import inspect

    stack = inspect.stack()

    for frame_info in stack[2:]:
        frame_locals = frame_info.frame.f_locals

        if "self" in frame_locals:
            cls = frame_locals["self"].__class__
            full_name = f"{cls.__module__}.{cls.__name__}"
            alias = SOURCE_ALIASES.get(full_name, full_name.split(".")[-1])
            return alias, cls.__name__

        if "cls" in frame_locals and isinstance(frame_locals["cls"], type):
            cls = frame_locals["cls"]
            full_name = f"{cls.__module__}.{cls.__name__}"
            alias = SOURCE_ALIASES.get(full_name, full_name.split(".")[-1])
            return alias, cls.__name__

        frame = frame_info.frame
        func_name = frame.f_code.co_name
        module_name = frame.f_globals.get("__name__", "")

        if module_name.startswith("iris_memory.") and not module_name.startswith(
            "iris_memory.stats"
        ):
            if module_name == "iris_memory.utils.llm_helper":
                continue

            module_short = module_name.split(".")[-1]
            alias = SOURCE_ALIASES.get(f"{module_name}.{func_name}", module_short)
            return alias, func_name

    return "unknown", "unknown"


def _record_stats(
    provider_id: str | None,
    result: LLMCallResult,
    duration_ms: float,
    prompt: str,
    is_multimodal: bool,
    image_count: int,
    source_module: str,
    source_class: str,
) -> None:
    """记录 LLM 调用统计（内部方法）"""
    try:
        import asyncio

        from iris_memory.stats import get_stats_registry

        registry = get_stats_registry()

        asyncio.create_task(
            registry.record_call(
                provider_id=provider_id,
                success=result.success,
                tokens_used=result.tokens_used,
                duration_ms=duration_ms,
                prompt=prompt,
                response=result.content,
                error=result.error if not result.success else None,
                is_multimodal=is_multimodal,
                image_count=image_count,
                source_module=source_module,
                source_class=source_class,
            )
        )
    except Exception as e:
        logger.debug(f"Failed to record stats: {e}")


def _extract_text(resp: Any) -> str:
    """从各种 LLM 响应形态中提取文本"""
    if hasattr(resp, "completion_text"):
        return (resp.completion_text or "").strip()
    if isinstance(resp, dict):
        return (resp.get("text", "") or resp.get("content", "")).strip()
    return str(resp).strip() if resp else ""
