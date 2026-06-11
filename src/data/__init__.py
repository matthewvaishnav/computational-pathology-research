"""Data namespace with lazy imports for optional slide-reading dependencies."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BioFormatsReader": ("src.data.wsi.format_support", "BioFormatsReader"),
    "UniversalSlideReader": ("src.data.wsi.format_support", "UniversalSlideReader"),
    "get_supported_formats": ("src.data.wsi.format_support", "get_supported_formats"),
    "open_slide": ("src.data.wsi.format_support", "open_slide"),
    "MultimodalDataset": ("src.data.loaders.loaders", "MultimodalDataset"),
    "collate_multimodal": ("src.data.loaders.loaders", "collate_multimodal"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a data component only when its public symbol is requested."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
