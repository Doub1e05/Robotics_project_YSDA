"""Small runtime patches for local mimic-video execution."""

from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    import numba
except Exception:  # pragma: no cover - best-effort runtime patch
    numba = None


_ORIGINAL_FILE_HANDLER = logging.FileHandler


class _RedirectingFileHandler(_ORIGINAL_FILE_HANDLER):
    """Redirect robosuite's hardcoded temp log to a writable local path."""

    def __init__(self, filename, *args, **kwargs):
        if filename == "/tmp/robosuite.log":
            target = os.environ.get("ROBOSUITE_LOG_PATH")
            if target:
                path = Path(target)
                path.parent.mkdir(parents=True, exist_ok=True)
                filename = str(path)
        super().__init__(filename, *args, **kwargs)


logging.FileHandler = _RedirectingFileHandler


if numba is not None:
    _ORIGINAL_NUMBA_JIT = numba.jit

    def _jit_without_cache(*args, **kwargs):
        kwargs["cache"] = False
        return _ORIGINAL_NUMBA_JIT(*args, **kwargs)

    numba.jit = _jit_without_cache
