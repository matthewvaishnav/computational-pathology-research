"""Training namespace with lazy imports for optional trainer dependencies."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "nnMILTrainer": (".nnmil_trainer", "nnMILTrainer"),
    "UnifiedTrainer": (".unified_trainer", "UnifiedTrainer"),
    "QuickTrainer": (".quick", "QuickTrainer"),
    "train": (".quick", "train"),
    "evaluate": (".quick", "evaluate"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a training component only when its public symbol is requested."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
