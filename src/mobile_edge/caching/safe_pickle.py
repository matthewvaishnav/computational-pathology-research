"""
Safe pickle deserialization with restricted class loading.

Prevents arbitrary code execution from malicious pickle data.
"""

import io
import pickle
from typing import Any


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows safe classes."""

    ALLOWED_MODULES = {
        "numpy.core.multiarray",
        "numpy",
        "numpy.ndarray",
        "builtins",
    }

    ALLOWED_CLASSES = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("builtins", "dict"),
        ("builtins", "list"),
        ("builtins", "tuple"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
    }

    def find_class(self, module: str, name: str):
        """Only allow safe classes to be unpickled."""
        if (module, name) in self.ALLOWED_CLASSES:
            return super().find_class(module, name)

        if module in self.ALLOWED_MODULES:
            return super().find_class(module, name)

        raise pickle.UnpicklingError(f"Forbidden class: {module}.{name}")


def safe_pickle_loads(data: bytes) -> Any:
    """Safely deserialize pickle data with restricted class loading."""
    return RestrictedUnpickler(io.BytesIO(data)).load()


def safe_pickle_load(file) -> Any:
    """Safely deserialize pickle data from file with restricted class loading."""
    return RestrictedUnpickler(file).load()
