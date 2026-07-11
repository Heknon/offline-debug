"""Tests for exceptions whose __init__ takes required keyword-only arguments."""

from __future__ import annotations

from io import BytesIO
from typing import Never, Self

from offline_debug import load_traceback, save_traceback

EXPECTED_CODE = 42


class KwargOnlyError(Exception):
    """Exception that only accepts a keyword-only argument and skips super().__init__."""

    def __init__(self, *, message: str) -> None:
        """Store the message without populating ``self.args`` via super().__init__."""
        self.message = message


class KwargGroup(ExceptionGroup):
    """Exception group subclass that requires an extra keyword-only argument."""

    code: int

    def __new__(cls, message: str, exceptions: list[Exception], *, code: int) -> Self:
        """Attach ``code`` on the group created by ``BaseExceptionGroup.__new__``."""
        self = super().__new__(cls, message, exceptions)
        self.code = code
        return self

    def __init__(self, message: str, exceptions: list[Exception], *, code: int) -> None:
        """Accept ``code`` so construction does not fall through to BaseException."""


def test_exception_kwargs_arguments() -> None:
    """
    A keyword-only exception should round-trip and reconstruct faithfully.

    Such an exception pickles but cannot be rebuilt by calling ``Cls(*self.args)``
    (its ``args`` is empty and ``message`` is required), so it exercises the
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


def test_exception_group_kwargs_arguments() -> None:
    """
    An ExceptionGroup subclass with a required keyword-only arg should round-trip.

    Default pickling reduces a group to ``(Cls, (message, exceptions), state)`` and
    rebuilds by calling ``Cls(message, exceptions)``, which fails when the subclass
    requires an extra keyword-only argument. Reconstruction must go through
    ``BaseExceptionGroup.__new__`` and restore the custom ``code`` state — without
    requiring the subclass to override ``derive``.
    """

    def raise_kwarg_group() -> Never:
        raise KwargGroup("grp", [ValueError("a"), KeyError("b")], code=EXPECTED_CODE)

    try:
        raise_kwarg_group()
    except KwargGroup as e:
        buffer = BytesIO()
        save_traceback(e, buffer)
        buffer.seek(0)
        loaded_exc = load_traceback(buffer, should_raise=False)

    assert isinstance(loaded_exc, KwargGroup)
    assert loaded_exc.code == EXPECTED_CODE
    assert loaded_exc.message == "grp"
    assert [type(inner).__name__ for inner in loaded_exc.exceptions] == [
        "ValueError",
        "KeyError",
    ]
