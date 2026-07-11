"""Tests for exceptions whose __init__ takes required keyword-only arguments."""

from __future__ import annotations

from io import BytesIO
from typing import Never

from offline_debug import load_traceback, save_traceback


class KwargOnlyError(Exception):
    """Exception that only accepts a keyword-only argument and skips super().__init__."""

    def __init__(self, *, message: str) -> None:
        """Store the message without populating ``self.args`` via super().__init__."""
        self.message = message


def test_exception_kwargs_arguments() -> None:
    """
    A keyword-only exception should round-trip and reconstruct faithfully.

    Such an exception pickles but cannot be rebuilt by calling ``Cls(*self.args)``
    (its ``args`` is empty and ``message`` is required), so it exercises the robust
    pickler that reconstructs exceptions via ``__new__``.
    """

    def raise_kwarg_error() -> Never:
        raise KwargOnlyError(message="hello")

    try:
        raise_kwarg_error()
    except KwargOnlyError as e:
        buffer = BytesIO()
        save_traceback(e, buffer)
        buffer.seek(0)
        loaded_exc = load_traceback(buffer, should_raise=False)

    assert isinstance(loaded_exc, KwargOnlyError)
    assert loaded_exc.message == "hello"
