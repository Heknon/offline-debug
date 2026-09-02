"""
A loaded exception must behave like a genuine one under introspection.

Reconstructed frames deliberately carry a *synthetic* code object, because
accessing ``f_locals`` on a frame built by ``PyFrame_New`` over real optimized
bytecode segfaults. The original ``tb_lasti`` indexes into the original
bytecode, so pairing it with that synthetic code object made
``traceback._get_code_position`` search a two-entry ``co_positions()`` for
instruction ``lasti // 2``. That raises StopIteration inside a generator, which
surfaced as ``RuntimeError: generator raised StopIteration`` from *any* attempt
to format a loaded exception -- the library's central promise.

Reconstructed frames now report no instruction offset, so the stdlib falls back
to ``tb_lineno``, which is restored accurately.
"""

from __future__ import annotations

import pickle
import traceback
from io import BytesIO
from typing import Never

from offline_debug import ExceptionData, load_traceback
from tests.helpers import roundtrip


def formatted(exc: BaseException) -> str:
    """Format an exception exactly as an unhandled traceback would print."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def positions(exc: BaseException) -> list[tuple[str, int | None, str]]:
    """Return ``(filename, lineno, function)`` for every frame of a traceback."""
    return [(f.filename, f.lineno, f.name) for f in traceback.extract_tb(exc.__traceback__)]


def inner() -> None:
    """Raise the exception under test."""
    local_value = 42
    raise ValueError(str(local_value))


def outer() -> None:
    """Add a second frame to the traceback."""
    inner()


def make_error() -> BaseException:
    """Return a ValueError carrying a two-function traceback."""
    try:
        outer()
    except ValueError as err:
        return err
    msg = "outer() did not raise"
    raise AssertionError(msg)


def test_loaded_exception_can_be_formatted() -> None:
    """Formatting a loaded exception must not raise."""
    restored = roundtrip(make_error())

    text = formatted(restored)

    assert "ValueError: 42" in text
    assert "Traceback (most recent call last):" in text


def test_loaded_traceback_keeps_original_frames() -> None:
    """
    The original frames survive verbatim as the tail of the loaded traceback.

    ``load_traceback`` splices the reconstructed frames onto the live stack so the
    exception appears to have been raised at the load site, so the loaded traceback
    is longer than the original by exactly the frames of the caller.
    """
    original = make_error()
    expected = positions(original)
    restored = roundtrip(original)

    actual = positions(restored)

    assert actual[-len(expected) :] == expected
    assert [name for _, _, name in expected] == ["make_error", "outer", "inner"]


def test_loaded_frames_report_source_lines() -> None:
    """Line numbers must be accurate enough for the stdlib to find the source."""
    restored = roundtrip(make_error())

    text = formatted(restored)

    assert "in inner" in text
    assert "raise ValueError(str(local_value))" in text


def test_chained_exception_keeps_cause_wording() -> None:
    """A loaded chain prints the same explanatory line as the original."""
    try:
        try:
            outer()
        except ValueError as err:
            msg = "wrapper"
            raise RuntimeError(msg) from err
    except RuntimeError as err:
        original = err

    assert "direct cause" in formatted(original)
    assert "direct cause" in formatted(roundtrip(original))


def test_self_caused_exception_can_be_formatted() -> None:
    """``raise e from e`` must format without looping, exactly as it does natively."""
    try:
        try:
            msg = "original"
            raise ValueError(msg)
        except ValueError as err:
            raise err from err
    except ValueError as err:
        original = err

    restored = roundtrip(original)
    assert restored.__cause__ is restored

    text = formatted(restored)

    assert "ValueError: original" in text


def test_exception_group_can_be_formatted() -> None:
    """A loaded group prints its sub-exceptions in the standard group layout."""
    try:
        raise ExceptionGroup("group", [ValueError("x"), TypeError("y")])
    except ExceptionGroup as err:
        original = err

    text = formatted(roundtrip(original))

    assert "ExceptionGroup: group (2 sub-exceptions)" in text
    assert "ValueError: x" in text
    assert "TypeError: y" in text


def test_suppressed_context_stays_suppressed() -> None:
    """``raise X from None`` must not print the context its author suppressed."""

    def raise_suppressing() -> Never:
        try:
            msg = "internal detail"
            raise KeyError(msg)
        except KeyError:
            msg = "public message"
            raise ValueError(msg) from None

    try:
        raise_suppressing()
    except ValueError as err:
        original = err

    restored = roundtrip(original)

    assert restored.__suppress_context__ is True
    assert "internal detail" not in formatted(restored)
    assert "During handling of the above exception" not in formatted(restored)


def test_implicit_context_is_still_shown() -> None:
    """Restoring suppression must not suppress an ordinary implicit context."""

    def raise_implicitly() -> Never:
        try:
            msg = "first failure"
            raise KeyError(msg)
        except KeyError:
            msg = "second failure"
            raise ValueError(msg)  # noqa: B904 - the implicit context is the point

    try:
        raise_implicitly()
    except ValueError as err:
        original = err

    restored = roundtrip(original)

    assert restored.__suppress_context__ is False
    assert "first failure" in formatted(restored)
    assert "During handling of the above exception" in formatted(restored)


def test_explicit_cause_keeps_suppression_flag() -> None:
    """``raise X from Y`` suppresses the context while still showing the cause."""

    def raise_from_cause() -> Never:
        try:
            msg = "root cause"
            raise KeyError(msg)
        except KeyError as err:
            msg = "wrapper"
            raise ValueError(msg) from err

    try:
        raise_from_cause()
    except ValueError as err:
        original = err

    restored = roundtrip(original)

    assert restored.__suppress_context__ is True
    assert "direct cause" in formatted(restored)


def test_suppression_survives_without_any_context() -> None:
    """``raise X from None`` outside an ``except`` block has no link to piggyback on."""
    original = ValueError("standalone")
    original.__suppress_context__ = True

    restored = roundtrip(original)

    assert restored.__cause__ is None
    assert restored.__context__ is None
    assert restored.__suppress_context__ is True


def test_dump_without_suppression_field_still_loads() -> None:
    """A pre-0.4.0 dump has no ``suppress_context`` key and must default to False."""
    data = ExceptionData(exc_pickle=pickle.dumps(ValueError("old")), tb_frames=[])
    # Reproduce the on-disk shape of a dump written before the field existed.
    del data.__dict__["suppress_context"]
    assert "suppress_context" not in data.__dict__

    buffer = BytesIO()
    pickle.dump(data, buffer)
    buffer.seek(0)
    restored = load_traceback(buffer, should_raise=False)

    assert isinstance(restored, ValueError)
    assert restored.__suppress_context__ is False
