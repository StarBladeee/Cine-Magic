"""
CineMagic 配置加载模块

Usage:
    from src.config import get_config
    cfg = get_config()
    data_dir = cfg['data']['douban_dir']
"""

import os
import yaml

_CONFIG = None


def get_config(config_path: str | None = None) -> dict:
    """
    加载并返回全局配置（单例模式）。

    Args:
        config_path: YAML配置文件路径，默认为项目根目录下的 config/config.yaml

    Returns:
        配置字典
    """
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    if config_path is None:
        # 自动定位到项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)

    _resolve_paths(_CONFIG, project_root)
    return _CONFIG


def _resolve_paths(config: dict, project_root: str) -> None:
    """将相对路径转为绝对路径（仅处理包含目录分隔符的值）"""
    for section in ["data", "output"]:
        if section in config:
            for key, value in config[section].items():
                if isinstance(value, str) and not os.path.isabs(value) and "/" in value:
                    config[section][key] = os.path.join(project_root, value)


def reload_config(config_path: str | None = None) -> dict:
    """强制重新加载配置"""
    global _CONFIG
    _CONFIG = None
    return get_config(config_path)
