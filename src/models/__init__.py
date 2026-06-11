"""Model namespace with lazy imports for optional research dependencies."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "PathoAlign": (".pathoalign", "PathoAlign"),
    "AttentionMIL": (".attention_mil", "AttentionMIL"),
    "CLAM": (".clam", "CLAM"),
    "CLAMModel": (".clam", "CLAMModel"),
    "GraphMIL": (".graph_mil", "GraphMIL"),
    "GraphMILClassifier": (".graph_mil", "GraphMILClassifier"),
    "GraphMILConfig": (".graph_mil", "GraphMILConfig"),
    "nnMIL": (".nnmil", "nnMIL"),
    "TransMIL": (".transmil", "TransMIL"),
    "TransMILModel": (".transmil_model", "TransMILModel"),
    "PatchCNN": (".patch_cnn", "PatchCNN"),
    "PCamPatchClassifier": (".patch_cnn", "PCamPatchClassifier"),
    "create_foundation_model": (".foundation", "create_foundation_model"),
    "get_foundation_model_info": (".foundation", "get_foundation_model_info"),
    "PathoAlignV2": (".pathoalign_v2", "PathoAlignV2"),
    "NNMILBaseline": (".baselines", "NNMILBaseline"),
    "GatedAttention": (".attention", "GatedAttention"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a model only when its public symbol is requested."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
