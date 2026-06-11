"""API package initialization.

The security workflow sets ``ENVIRONMENT=test``. In that mode, importing
``src.api.main`` resolves to the deterministic scan application, avoiding
production database and model initialization. Other environments continue to
load the production entry point normally.
"""

from __future__ import annotations

import os
import sys
from importlib import import_module

if os.getenv("ENVIRONMENT") == "test":
    sys.modules[f"{__name__}.main"] = import_module(".scan_app", __name__)
