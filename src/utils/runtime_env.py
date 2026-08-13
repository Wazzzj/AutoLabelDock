"""Process-level safeguards for native GUI and training dependencies."""
from __future__ import annotations

import os
import site
import sys
from pathlib import Path


def disable_user_site_packages() -> list[str]:
    """Remove per-user packages so one process never mixes two Python environments."""
    os.environ["PYTHONNOUSERSITE"] = "1"
    configured = site.getusersitepackages()
    user_sites = [configured] if isinstance(configured, str) else list(configured)
    normalized = {
        os.path.normcase(os.path.normpath(str(Path(path))))
        for path in user_sites
    }
    removed = [
        path
        for path in sys.path
        if os.path.normcase(os.path.normpath(path)) in normalized
    ]
    sys.path[:] = [
        path
        for path in sys.path
        if os.path.normcase(os.path.normpath(path)) not in normalized
    ]
    site.ENABLE_USER_SITE = False
    return removed


def configure_headless_matplotlib() -> str:
    """Force a non-GUI backend before Ultralytics creates training plots."""
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg", force=True)
    return str(matplotlib.get_backend())
