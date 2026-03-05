"""
配置加载链 — 多层配置源合并

优先级（由高到低）：
  Level 1: 用户简单配置 (_conf_schema.json 对应的 AstrBot config 对象) — 只读
  Level 2: 插件数据配置 (Plugin Data, JSON 持久化) — 可读写
  Level 3: Schema 默认值

合并规则：高优先级非 None 值覆盖低优先级。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from iris_memory.config.schema import SCHEMA, AccessLevel
from iris_memory.config.validators import inject_defaults, validate_dict, ConfigValidationError

logger = logging.getLogger(__name__)

# 持久化文件名
PLUGIN_DATA_FILENAME = "iris_config.json"


class ConfigLoader:
    """配置加载器"""

    def __init__(
        self,
        user_config: Any = None,
        plugin_data_path: Optional[Path] = None,
    ):
        self._user_config = user_config
        self._plugin_data_path = plugin_data_path

    @property
    def plugin_data_file(self) -> Optional[Path]:
        if self._plugin_data_path is None:
            return None
        return self._plugin_data_path / PLUGIN_DATA_FILENAME

    def load(self) -> Dict[str, Any]:
        """执行完整加载链，返回合并后的配置字典

        Returns:
            扁平化的 ``{key: value}`` 配置字典
        """
        # 1. 从 Schema 获取全量默认值
        merged = {key: field.default for key, field in SCHEMA.items()}

        # 2. Level 2: 从持久化文件加载可写配置
        level2 = self._load_plugin_data()
        for key, val in level2.items():
            if key in SCHEMA:
                merged[key] = val

        # 3. Level 1: 从用户配置加载（最高优先级）
        level1 = self._extract_user_config()
        for key, val in level1.items():
            if val is not None and key in SCHEMA:
                merged[key] = val

        # 4. 校验并注入缺失默认值
        try:
            merged = validate_dict(merged)
        except ConfigValidationError as exc:
            logger.warning("配置校验有错误（已降级到默认值）: %s", exc)

        return merged

    def _extract_user_config(self) -> Dict[str, Any]:
        """从 AstrBot 用户配置对象中提取扁平化的配置值

        遍历 Schema 中 AccessLevel.READONLY 的键，
        从 ``user_config`` 按 ``section.attr`` 路径读取。
        """
        if self._user_config is None:
            return {}

        result: Dict[str, Any] = {}
        for key, field in SCHEMA.items():
            value = self._get_nested(self._user_config, key)
            if value is not None:
                result[key] = value
        return result

    def _load_plugin_data(self) -> Dict[str, Any]:
        """从持久化 JSON 文件加载 Level 2 配置"""
        fp = self.plugin_data_file
        if fp is None or not fp.exists():
            return {}
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            # 展开嵌套结构
            return self._flatten(data)
        except (json.JSONDecodeError, OSError):
            logger.exception("加载插件持久化配置失败: %s", fp)
            return {}

    @staticmethod
    def _get_nested(obj: Any, dotted_key: str) -> Any:
        """从嵌套对象/字典按点号路径读取值"""
        parts = dotted_key.split(".")
        current = obj
        for part in parts:
            if current is None:
                return None
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    @staticmethod
    def _flatten(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """将嵌套字典展平为 ``section.key`` 格式"""
        result: Dict[str, Any] = {}
        for k, v in data.items():
            full_key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                result.update(ConfigLoader._flatten(v, full_key))
            else:
                result[full_key] = v
        return result
