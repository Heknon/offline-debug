"""Helpers shared by the test modules."""

from __future__ import annotations

from io import BytesIO

from offline_debug import load_traceback, save_traceback


def roundtrip(exc: BaseException) -> BaseException:
    """Save an exception to a buffer and load it back without raising."""
    buffer = BytesIO()
    save_traceback(exc, buffer)
    buffer.seek(0)
    return load_traceback(buffer, should_raise=False)
