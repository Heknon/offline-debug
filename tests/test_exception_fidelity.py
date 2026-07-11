"""
Regression tests: well-behaved exceptions must keep their vanilla pickle fidelity.

The custom pickler only takes over reconstruction for exceptions whose
Python-defined constructors can reject ``Cls(*args)`` at load time. Everything
else — builtin exceptions with C-level state, classes with their own pickle
protocol hooks — must round-trip exactly as vanilla pickle would.
"""

from __future__ import annotations

from io import BytesIO
from typing import Never, Self

from offline_debug import load_traceback, save_traceback

EXPECTED_ERRNO = 2
EXPECTED_VALUE = 42


def roundtrip(exc: BaseException) -> BaseException:
    """Save an exception to a buffer and load it back without raising."""
    buffer = BytesIO()
    save_traceback(exc, buffer)
    buffer.seek(0)
    return load_traceback(buffer, should_raise=False)


class SetstateError(Exception):
    """Exception that relies on ``__setstate__`` to finish reconstruction."""

    def __init__(self, message: str) -> None:
        """Store an instance attribute so the pickle carries state to restore."""
        super().__init__(message)
        self.message = message

    def __setstate__(self, state: dict[str, object] | None, /) -> None:
        """Restore state and mark that the pickle protocol hook actually ran."""
        self.__dict__.update(state or {})
        self.restored = True


class SlottedError(Exception):
    """Exception whose extra state lives in ``__slots__``, not ``__dict__``."""

    __slots__ = ("code",)

    def __init__(self, code: int) -> None:
        """Populate ``args`` via super() and keep ``code`` in a slot."""
        super().__init__(code)
        self.code = code


class FrozenArgsError(Exception):
    """Exception with a read-only ``args`` and a required keyword-only argument."""

    args = property(lambda _self: ())

    def __init__(self, *, message: str) -> None:
        """Store the message without populating ``self.args``."""
        self.message = message


class CodedGroup(ExceptionGroup):
    """Exception group subclass carrying extra state, with the default derive."""

    code: int

    def __new__(cls, message: str, exceptions: list[Exception], *, code: int) -> Self:
        """Attach ``code`` on the group created by ``BaseExceptionGroup.__new__``."""
        self = super().__new__(cls, message, exceptions)
        self.code = code
        return self

    def __init__(self, message: str, exceptions: list[Exception], *, code: int) -> None:
        """Accept ``code`` so construction does not fall through to BaseException."""


def test_oserror_attributes_preserved() -> None:
    """OSError state lives at the C level and is restored by its own __reduce__."""

    def raise_oserror() -> Never:
        raise OSError(EXPECTED_ERRNO, "No such file or directory", "missing.txt")

    try:
        raise_oserror()
    except OSError as e:
        loaded_exc = roundtrip(e)

    assert isinstance(loaded_exc, FileNotFoundError)
    assert loaded_exc.errno == EXPECTED_ERRNO
    assert loaded_exc.strerror == "No such file or directory"
    assert loaded_exc.filename == "missing.txt"


def test_stopiteration_value_preserved() -> None:
    """StopIteration.value is C-level state populated by the builtin __init__."""

    def raise_stopiteration() -> Never:
        raise StopIteration(EXPECTED_VALUE)

    try:
        raise_stopiteration()
    except StopIteration as e:
        loaded_exc = roundtrip(e)

    assert isinstance(loaded_exc, StopIteration)
    assert loaded_exc.value == EXPECTED_VALUE


def test_custom_setstate_honored() -> None:
    """An exception's own pickle protocol hooks must not be bypassed."""

    def raise_setstate_error() -> Never:
        raise SetstateError("hello")

    try:
        raise_setstate_error()
    except SetstateError as e:
        loaded_exc = roundtrip(e)

    assert isinstance(loaded_exc, SetstateError)
    assert loaded_exc.restored is True


def test_slotted_exception_preserved() -> None:
    """
    Slot state set by a Python __init__ must survive the round-trip.

    The takeover path applies (Python-defined __init__), so reconstruction must
    retry normal construction — a bare ``__new__`` + ``__dict__`` restore would
    silently drop the slot.
    """

    def raise_slotted() -> Never:
        raise SlottedError(EXPECTED_VALUE)

    try:
        raise_slotted()
    except SlottedError as e:
        loaded_exc = roundtrip(e)

    assert isinstance(loaded_exc, SlottedError)
    assert loaded_exc.code == EXPECTED_VALUE


def test_readonly_args_exception() -> None:
    """
    A read-only ``args`` must not abort reconstruction.

    The keyword-only ``message`` forces the ``__new__`` fallback, where assigning
    ``args`` fails on the property; reconstruction must degrade gracefully and
    still restore the instance state.
    """

    def raise_frozen_args() -> Never:
        raise FrozenArgsError(message="hello")

    try:
        raise_frozen_args()
    except FrozenArgsError as e:
        loaded_exc = roundtrip(e)

    assert isinstance(loaded_exc, FrozenArgsError)
    assert loaded_exc.message == "hello"


def test_exception_group_subclass_without_derive() -> None:
    """
    A group subclass must survive load without overriding ``derive``.

    The load path rebuilds groups around their reconstructed inner exceptions;
    relying on the default ``derive`` would return a plain ExceptionGroup and
    silently drop the subclass type and its custom state.
    """

    def raise_coded_group() -> Never:
        raise CodedGroup("grp", [ValueError("a")], code=EXPECTED_VALUE)

    try:
        raise_coded_group()
    except CodedGroup as e:
        loaded_exc = roundtrip(e)

    assert isinstance(loaded_exc, CodedGroup)
    assert loaded_exc.code == EXPECTED_VALUE
