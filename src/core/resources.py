"""Static resource path loader backed by config/static_resources.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

APP_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_MODEL_DIR = APP_ROOT / "pretrained_models"

STATIC_RESOURCE_CONFIG_PATH = APP_ROOT / "config" / "static_resources.json"
_MISSING_RESOURCE = APP_ROOT / "__missing_resource__"


def _load_resource_config() -> dict[str, Any]:
    """Load static resource path settings from JSON."""
    if not STATIC_RESOURCE_CONFIG_PATH.exists():
        logger.warning("Static resource config not found: %s", STATIC_RESOURCE_CONFIG_PATH)
        return {}
    try:
        data = json.loads(STATIC_RESOURCE_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Failed to load static resource config %s: %s",
            STATIC_RESOURCE_CONFIG_PATH,
            exc,
        )
        return {}
    if not isinstance(data, dict):
        logger.warning("Static resource config must be a JSON object: %s", STATIC_RESOURCE_CONFIG_PATH)
        return {}
    return data


def _as_path(value: object) -> Path:
    """Resolve a JSON path value relative to the application root."""
    if not isinstance(value, str) or not value.strip():

        return _MISSING_RESOURCE
    path = Path(value)
    if path.is_absolute():
        return path
    return APP_ROOT / path


def _path_list(values: object) -> tuple[Path, ...]:
    """Resolve a JSON string list into a tuple of Paths."""
    if not isinstance(values, list):
        return tuple()
    result: list[Path] = []
    for item in values:
        if isinstance(item, str) and item.strip():
            result.append(_as_path(item))
    return tuple(result)



_RESOURCE_CONFIG = _load_resource_config()
_DIRECTORIES = _RESOURCE_CONFIG.get("directories", {})
_ICONS = _RESOURCE_CONFIG.get("icons", {})
_SCREENSHOT_CONFIG = _RESOURCE_CONFIG.get("screenshots", {})

ICON_DIR = _as_path(_DIRECTORIES.get("icon") if isinstance(_DIRECTORIES, dict) else None)
LOGO_DIR = _as_path(_DIRECTORIES.get("logo") if isinstance(_DIRECTORIES, dict) else None)
RESOURCE_DIR = _as_path(_DIRECTORIES.get("resources") if isinstance(_DIRECTORIES, dict) else None)
SCREENSHOT_DIR = _as_path(_DIRECTORIES.get("screenshots") if isinstance(_DIRECTORIES, dict) else None)

APP_LOGO_CANDIDATES = _path_list(_RESOURCE_CONFIG.get("app_logo_candidates"))

LOADING_SVG = _as_path(_ICONS.get("loading") if isinstance(_ICONS, dict) else None)
TREE_CLOSED_SVG = _as_path(_ICONS.get("tree_closed") if isinstance(_ICONS, dict) else None)
TREE_OPEN_SVG = _as_path(_ICONS.get("tree_open") if isinstance(_ICONS, dict) else None)

SCREENSHOTS = {
    key: _as_path(value)
    for key, value in (
        _SCREENSHOT_CONFIG.items() if isinstance(_SCREENSHOT_CONFIG, dict) else []
    )
}


def stylesheet_url(path: Path) -> str:
    """Return a path string suitable for Qt stylesheet url(...)."""
    return path.as_posix()


def resolve_pretrained_model_path(value: str | Path) -> Path:
    """Resolve bundled/repository pretrained weights without blocking downloads."""
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path

    candidates: list[Path] = []
    if path.parent == Path("."):
        candidates.append(PRETRAINED_MODEL_DIR / path.name)
    else:
        candidates.append(APP_ROOT / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return path
